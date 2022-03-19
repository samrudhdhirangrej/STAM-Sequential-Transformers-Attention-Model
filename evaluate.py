import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
import random

from torch import optim
from timm.models import create_model
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

from datasets import build_dataset
import models
import utils
import json

from paths import OUTPUTROOT
import fire

import torch
from timm.utils import accuracy

@torch.no_grad()
def evaluate(data_loader, model, device, output_dir):

    metric_logger = utils.MetricLogger(output_dir, delimiter="  ")
    header = 'Test:'

    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, _, output_dist, _ = model(images, target, None, None)

        batch_size = images.shape[0]
        for i in range(output.size(0)):
            accg = accuracy(output[i], target, topk=(1,))[0]
            accd = accuracy(output_dist[i], target, topk=(1,))[0]
            accf = accuracy(torch.softmax(output[i],dim=-1)+torch.softmax(output_dist[i],dim=-1), target, topk=(1,))[0]
            metric_logger.meters['accGT_T'+str(i)].update(accg.item(), n=batch_size)
            metric_logger.meters['accDT_T'+str(i)].update(accd.item(), n=batch_size)
            metric_logger.meters['accFS_T'+str(i)].update(accf.item(), n=batch_size)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(batch_size_val=1000, trainmaxT=21, maxT=49, num_workers=8, model_name=None, pin_mem=True, drop=0.0, drop_path=0.1, checkpoint_filename=None, output_dir='train', epochs=10, dataset='imagenet', pretrained=True, input_size=224, sync_bn=True, mlp_layers=4, mlp_hidden_dim=2048):

    device = torch.device('cuda')
    output_dir = OUTPUTROOT / dataset / output_dir
    if not (output_dir).exists():
        (output_dir).mkdir()

    ''' fix the seed for reproducibility '''
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    ''' dataloader '''
    dataset_val, nb_classes = build_dataset(dataset=dataset, is_train=False, input_size=input_size, erasing_aug=False)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size_val,
        num_workers=num_workers,
        pin_memory=pin_mem,
        worker_init_fn=utils.seed_worker,
        generator=torch.Generator(),
        drop_last=False
    )

    model = create_model(
        model_name,
        pretrained=pretrained, 
        num_classes=nb_classes,
        drop_rate=drop,
        drop_path_rate=drop_path,
        drop_block_rate=None,
    )
    model.set_mode(trainmaxT, 1, mlp_layers, mlp_hidden_dim)
    model.T = maxT
    model.classifier_criterion = torch.nn.CrossEntropyLoss()
    model.dist_criterion = lambda x, y, z: torch.zeros(1).mean().to(x.device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.eval()

    '''load model checkpoint from checkpoint_train'''
    checkpoint = torch.load(PRETRAINED+checkpoint_filename, map_location=device)
    model.load_state_dict(checkpoint['model'])

    #############################################################
    ''''''''''''''''''''' Starting the loop '''''''''''''''''''''

    AccGT, AccDT, AccFS = [], [], []
    for seed in range(epochs):
        with (output_dir / "log.txt").open("a") as f:
            f.write(f"seed: {seed} \n")

        ''' fix the seed for reproducibility '''
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        test_stats = evaluate(data_loader_val, model, device, output_dir)

        AccGT += [np.array([test_stats['accGT_T'+str(i)] for i in range(maxT)])]
        AccDT += [np.array([test_stats['accDT_T'+str(i)] for i in range(maxT)])]
        AccFS += [np.array([test_stats['accFS_T'+str(i)] for i in range(maxT)])]

        checkpoint_path = output_dir / 'final_test_seed_'+str(seed)+'.pth'
        utils.save_on_master({
            'test_AccGT': AccGT[seed],
            'test_AccDT': AccDT[seed],
            'test_AccFS': AccFS[seed],
            }, checkpoint_path)

if __name__ == '__main__':
    fire.Fire(main)




