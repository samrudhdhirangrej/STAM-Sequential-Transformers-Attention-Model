'''
based on: https://github.com/facebookresearch/deit
'''

import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import random

from torch import optim
from timm.models import create_model
from torch.utils.tensorboard import SummaryWriter
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from scheduler_factory import create_scheduler
from timm.optim.optim_factory import create_optimizer_v2
from timm.utils import NativeScaler 
from torch.nn import functional as F

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from samplers import RandomSubsetSampler, DistributedSamplerWrapper
import models
import utils
import json

from paths import OUTPUTROOT, LOGFOLDER
import fire

def main(other_lr=5e-4, critic_lr = 0, seed=0, batch_size=64*8, batch_size_val=1000, maxT=16, stepT=1, num_workers=8, model_name='RAM_deit_tiny_patch16_224', teacher_name=None, pin_mem=True, smoothing=False, drop=0.0, drop_path=0.1, output_dir='train', epochs=100, dataset='imagenet', clip_grad=None, training_mode=True, world_size=4, distributed=False, pretrained=True, mixup=False, erasing=False, input_size=224, min_lr=1e-5, weight_decay=0.05, dist_eval=True, sync_bn=True, loc_tau=4, mlp_layers=4, mlp_hidden_dim=2048, dist_type='soft', dist_temp=1, checkpoint_epoch=None):

    ''' loggging '''
    params = locals().copy()

    if distributed:
        gpu = utils.init_distributed_mode(world_size)
    device = torch.device('cuda')

    log_dir = LOGFOLDER/ output_dir
    output_dir = OUTPUTROOT / dataset / output_dir
    if utils.is_main_process():
        if not (output_dir).exists():
            (output_dir).mkdir()
        with open(output_dir/'args.json','a+') as f:
            json.dump(params,f)

    ''' fix the seed for reproducibility '''
    seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    ''' dataloader '''
    dataset_train, nb_classes = build_dataset(dataset=dataset, is_train=True, input_size=input_size, erasing_aug=erasing)
    dataset_val, nb_classes = build_dataset(dataset=dataset, is_train=False, input_size=input_size, erasing_aug=False)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    if distributed:
        sampler_train = RandomSubsetSampler(dataset_train, generator=torch.Generator(), subset_samples=(len(dataset_train)//(maxT-1)))
        sampler_train = DistributedSamplerWrapper(sampler_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if dist_eval: sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = RandomSubsetSampler(dataset_train, generator=torch.Generator(), subset_samples=(len(dataset_train)//(maxT-1)))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        worker_init_fn=utils.seed_worker,
        generator=torch.Generator(),
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size_val,
        num_workers=num_workers,
        pin_memory=pin_mem,
        worker_init_fn=utils.seed_worker,
        generator=torch.Generator(),
        drop_last=False
    )

    ''' model definition '''
    teacher = create_model(
        teacher_name,
        pretrained=pretrained, 
        num_classes=nb_classes,
        drop_rate=drop,
        drop_path_rate=drop_path,
        drop_block_rate=None,
    )
    teacher = teacher.to(device)
    teacher.eval()

    model = create_model(
        model_name,
        pretrained=pretrained, 
        num_classes=nb_classes,
        drop_rate=drop,
        drop_path_rate=drop_path,
        drop_block_rate=None,
    )
    model.set_mode(maxT, stepT, mlp_layers, mlp_hidden_dim)
    model.to(device)

    ''' distibuted initi '''

    if distributed and sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        if teacher is not None:
            teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[gpu], find_unused_parameters=True)
            teacher = teacher.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with (output_dir / "log.txt").open("a") as f:
        f.write('number of params: {} \n'.format(n_parameters))


    ############################################################
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ############################################################
    ''''''''''''''''''''''' Training '''''''''''''''''''''''''''
    ############################################################
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    if utils.is_main_process():
        logger = SummaryWriter(log_dir /'log_' + datetime.datetime.now().isoformat(sep='-'))

    if mixup:
        # hard code values from DeiT repository
        mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=nb_classes)
    else:
        mixup_fn = None

    if mixup:
        # smoothing is handled with mixup label transform
        classifier_criterion = SoftTargetCrossEntropy()
    elif smoothing:
        classifier_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        classifier_criterion = torch.nn.CrossEntropyLoss()

    if dist_type=='hard':
        balance=0.5
        dist_criterion = lambda x, y, z: F.cross_entropy(x, (torch.softmax(y, dim=-1) + torch.softmax(z, dim=-1)).argmax(-1))
    elif dist_type=='soft':
        balance=0.5
        dist_criterion = utils.Dist_KLD(dist_temp)
    else:
        balance=0.0
        dist_criterion = lambda x, y, z: torch.zeros(1).mean().to(x.device)

    other_lr = other_lr * batch_size * utils.get_world_size() / 512.0
    critic_lr = critic_lr * batch_size * utils.get_world_size() / 512.0

    model_without_ddp.classifier_criterion = classifier_criterion
    model_without_ddp.dist_criterion = dist_criterion
    params = utils.separate_components_and_add_weight_decay(model_without_ddp, weight_decay, {'other':other_lr, 'critic':critic_lr})
    optimizer = optim.AdamW(params)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(optimizer, epochs, sched='cosine', min_lr=min_lr)
    start_epoch = 0
    max_accuracy = 0.0

    if checkpoint_epoch is not None:
        with (output_dir / "log.txt").open("a") as f:
            f.write("resuming from epoch "+str(checkpoint_epoch) + "\n")

        '''load model checkpoint from checkpoint_train'''
        checkpoint_path = output_dir / 'checkpoint_train_'+str(checkpoint_epoch)+'.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

        '''load random state checkpoint from checkpoint_test'''
        checkpoint_path = output_dir / 'seed_test_'+str(checkpoint_epoch)+'_rank_'+str(utils.get_rank())+'.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
        torch.random.set_rng_state(checkpoint['torch_random_rng_state'].cpu())
        random.setstate(checkpoint['random_rng_state'])
        np.random.set_state(checkpoint['np_random_rng_state'])
        checkpoint_path = output_dir / 'checkpoint_test_'+str(checkpoint_epoch)+'.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        max_accuracy = checkpoint['max_accuracy']


    with (output_dir / "log.txt").open("a") as f:
        f.write(f"Start training for {epochs} epochs \n")
    start_time = time.time()

    #############################################################
    ''''''''''''''''''''' Starting the loop '''''''''''''''''''''

    for epoch in range(start_epoch, epochs):

        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write('epoch started at '+datetime.datetime.now().isoformat(sep='-')+'\n')

        #########################################################
        ''''''''''''''''''''''''' Train '''''''''''''''''''''''''

        if distributed:
            data_loader_train.sampler.set_epoch(epoch)
            data_loader_train.generator.manual_seed(utils.get_rank()*epochs + epoch)
        else:
            data_loader_train.sampler.generator.manual_seed(5728479885 + epoch)
            data_loader_train.generator.manual_seed(5728479885 + epoch)

        if utils.is_main_process():
            logger.add_scalar('learning_rate/actor', optimizer.param_groups[0]['lr'], epoch)
            logger.add_scalar('learning_rate/critic', optimizer.param_groups[1]['lr'], epoch)

        model_without_ddp.loc_tau = min(1 + ((loc_tau-1)*epoch/99), loc_tau)

        train_stats = train_one_epoch(
            model, teacher, output_dir, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            clip_grad, mixup_fn,
            set_training_mode=training_mode,
            balance=balance
        )

        if utils.is_main_process():
            logger.add_scalar('Acc/train_acc1', train_stats['acc1'], epoch)
            logger.add_scalar('Loss/train_cls', train_stats['cls_loss'], epoch)
            logger.add_scalar('Loss/train_actor', train_stats['actor_loss'], epoch)
            logger.add_scalar('Loss/train_critic', train_stats['critic_loss'], epoch)
            logger.add_scalar('Loss/train_dist', train_stats['dist_loss'], epoch)

        lr_scheduler.step(epoch)
        checkpoint_path = output_dir / 'checkpoint_train_'+str(epoch)+'.pth'
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),},
            checkpoint_path)

        checkpoint_path = output_dir / 'seed_train_'+str(epoch)+'_rank_'+str(utils.get_rank())+'.pth'
        save_dir = {
            'torch_random_rng_state': torch.random.get_rng_state().cuda(),
            'cuda_rng_state': torch.cuda.get_rng_state().cuda(),
            'random_rng_state': random.getstate(),
            'np_random_rng_state': np.random.get_state(),}
        torch.save(save_dir, checkpoint_path)

        with (output_dir / "log.txt").open("a") as f:
            f.write('epoch [{}] lr: actor {:.6f} critic {:.6f}, loc_tau {:.6f}, balance {:.1f} \n'.format(epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr'], model_without_ddp.loc_tau, balance))

        ########################################################
        ''''''''''''''''''''''''' Test '''''''''''''''''''''''''
        test_stats = evaluate(data_loader_val, model, teacher, device, output_dir)

        if utils.is_main_process():
            logger.add_scalar('Acc/test_accGT', test_stats['accGT'], epoch)
            logger.add_scalar('Acc/test_accDT', test_stats['accDT'], epoch)
            logger.add_scalar('Acc/test_accFS', test_stats['accFS'], epoch)
            logger.add_scalar('Loss/test_cls', test_stats['loss'], epoch)
            logger.add_scalar('Loss/test_dist', test_stats['dist_loss'], epoch)
            for i in range(maxT//stepT):
                logger.add_scalar('AccGT/test_acc_'+str(i), test_stats['accGT_T'+str(i)], epoch)
                logger.add_scalar('AccDT/test_acc_'+str(i), test_stats['accDT_T'+str(i)], epoch)
                logger.add_scalar('AccFS/test_acc_'+str(i), test_stats['accFS_T'+str(i)], epoch)

        max_accuracy = max(max_accuracy, test_stats["accGT"])
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(test_stats) + "\n")
            f.write(f'Max accuracy: {max_accuracy:.2f}% \n')
            f.write(json.dumps(log_stats) + "\n")
            f.write("\n\n")

        checkpoint_path = output_dir / 'checkpoint_test_'+str(epoch)+'.pth'
        utils.save_on_master({
            'epoch': epoch,
            'max_accuracy': max_accuracy,
            'test_accGT': test_stats["accGT"],
            'test_accDT': test_stats["accDT"],
            'test_accFS': test_stats["accFS"],},
            checkpoint_path)

        checkpoint_path = output_dir / 'seed_test_'+str(epoch)+'_rank_'+str(utils.get_rank())+'.pth'
        save_dir = {
            'torch_random_rng_state': torch.random.get_rng_state().cuda(),
            'cuda_rng_state': torch.cuda.get_rng_state().cuda(),
            'random_rng_state': random.getstate(),
            'np_random_rng_state': np.random.get_state(),
            }
        torch.save(save_dir, checkpoint_path)

        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write('epoch ended at '+datetime.datetime.now().isoformat(sep='-')+'\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    with (output_dir / "log.txt").open("a") as f:
        f.write('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    fire.Fire(main)
