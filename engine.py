"""
modified starting from: https://github.com/facebookresearch/deit
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
#from torchvision import utils as vutils

from time import time
from DeiTViT import DeiTVisionTransformer

def train_one_epoch(model: torch.nn.Module,
                    teacher: torch.nn.Module,
                    output_dir,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, balance=0.5):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(output_dir, delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            if teacher is not None:
                if isinstance(teacher, DeiTVisionTransformer):
                    teacher_gt, teacher_dist = teacher(samples)
                else:
                    teacher_gt = teacher(samples)
                    teacher_dist = teacher_gt
            else:
                teacher_gt, teacher_dist = None, None

        if utils.is_dist_avail_and_initialized():
            model_without_ddp = model.module
        else:
            model_without_ddp = model

        if mixup_fn is not None:
            samples, targets_smooth = mixup_fn(samples, targets)
        else:
            targets_smooth = targets

        with torch.cuda.amp.autocast():
            model_without_ddp.iter_init(samples, targets_smooth)

        cls_loss_value = []
        actor_loss_value = []
        critic_loss_value = []
        dist_loss_value = []
        while model_without_ddp.tempT != model_without_ddp.T:

            with torch.cuda.amp.autocast():
                cls_loss, actor_loss, critic_loss, dist_loss = model(samples, targets_smooth, teacher_gt, teacher_dist)
                outputs = model_without_ddp.return_logits
   
            loss = (1-balance)*cls_loss + actor_loss + critic_loss + dist_loss*balance
            cls_loss_value.append(cls_loss.item())
            dist_loss_value.append(dist_loss.item())
            if model_without_ddp.tempT < model_without_ddp.T:
                actor_loss_value.append(actor_loss.item())
                critic_loss_value.append(critic_loss.item())
    
            if not math.isfinite(loss.item()):
                print("cls actor critic losses are {} {} {}, stopping training".format(cls_loss.item(), actor_loss.item(), critic_loss.item()))
                sys.exit(1)
    
            optimizer.zero_grad()
    
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
    
            torch.cuda.synchronize()

        cls_loss_value = sum(cls_loss_value)/len(cls_loss_value)
        dist_loss_value = sum(dist_loss_value)/len(dist_loss_value)
        actor_loss_value = sum(actor_loss_value)/len(actor_loss_value)
        critic_loss_value = sum(critic_loss_value)/len(critic_loss_value)

        if outputs.dim()==3:
            targets = targets.unsqueeze(0).repeat([outputs.size(0)]+[1]*targets.dim()).flatten(0,1)
            outputs = outputs.flatten(0,1)
            
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.update(cls_loss=cls_loss_value)
        metric_logger.update(dist_loss=dist_loss_value)
        metric_logger.update(actor_loss=actor_loss_value)
        metric_logger.update(critic_loss=critic_loss_value)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    with (output_dir / "log.txt").open("a") as f:
        f.write("Averaged stats:")
        f.write(str(metric_logger))
        f.write('\n\n')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, teacher, device, output_dir):

    metric_logger = utils.MetricLogger(output_dir, delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if teacher is not None:
            if isinstance(teacher, DeiTVisionTransformer):
                teacher_gt, teacher_dist = teacher(images)
            else:
                teacher_gt = teacher(images)
                teacher_dist = teacher_gt
        else:
            teacher_gt, teacher_dist = None, None

        # compute output
        with torch.cuda.amp.autocast():
            output, loss, output_dist, loss_dist = model(images, target, teacher_gt, teacher_dist)

        torch.cuda.synchronize()

        batch_size = images.shape[0]
        all_accg, all_accd, all_accf = [], [], []
        for i in range(output.size(0)):
            accg = accuracy(output[i], target, topk=(1,))[0]
            accd = accuracy(output_dist[i], target, topk=(1,))[0]
            accf = accuracy(torch.softmax(output[i],dim=-1)+torch.softmax(output_dist[i],dim=-1), target, topk=(1,))[0]
            metric_logger.meters['accGT_T'+str(i)].update(accg.item(), n=batch_size)
            metric_logger.meters['accDT_T'+str(i)].update(accd.item(), n=batch_size)
            metric_logger.meters['accFS_T'+str(i)].update(accf.item(), n=batch_size)
            all_accg.append(accg)
            all_accd.append(accd)
            all_accf.append(accf)
        accg = sum(all_accg)/len(all_accg)
        accd = sum(all_accd)/len(all_accd)
        accf = sum(all_accf)/len(all_accf)

        metric_logger.update(loss=loss.item())
        metric_logger.update(dist_loss=loss_dist.item())
        metric_logger.meters['accGT'].update(accg.item(), n=batch_size)
        metric_logger.meters['accDT'].update(accd.item(), n=batch_size)
        metric_logger.meters['accFS'].update(accf.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    with (output_dir / "log.txt").open("a") as f:
        f.write('* Acc@GT {accg.global_avg:.3f} Acc@DT {accd.global_avg:.3f} Acc@FS {accf.global_avg:.3f} cls_loss {losses.global_avg:.3f} dist_loss {dist_loss.global_avg:.3f}\n\n'
          .format(accg=metric_logger.accGT, accd=metric_logger.accDT, accf=metric_logger.accFS, losses=metric_logger.loss, dist_loss=metric_logger.dist_loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



