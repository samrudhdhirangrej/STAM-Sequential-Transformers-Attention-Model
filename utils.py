"""
adapted from: https://github.com/facebookresearch/deit
Misc functions, including distributed helpers.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

import numpy as np
import random
from torch.nn import functional as F

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{global_avg:.4f}"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, output_dir, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_dir = output_dir

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'rank '+str(get_rank()),
            'eta: {eta}',
            '{meters}',
        ]
        log_msg.append('\n')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self)))
                else:
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        with (self.output_dir / "log.txt").open("a") as f:
            f.write('{} Total time: {} ({:.4f} s / it) \n'.format(
                header, total_time_str, total_time / len(iterable)))

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(world_size):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        distributed = False
        return

    distributed = True

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    torch.distributed.init_process_group(backend=dist_backend, init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    return gpu

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def separate_components_and_add_weight_decay(model, weight_decay, lr, skip_list=()):
    decay = {k:[] for k in lr.keys()}
    no_decay = {k:[] for k in lr.keys()}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            added = False
            for k in lr.keys():
                if k in name:
                    no_decay[k].append(param)
                    added = True
            if not added:
                no_decay['other'].append(param)
        else:
            added = False
            for k in lr.keys():
                if k in name:
                    decay[k].append(param)
                    added = True
            if not added:
                decay['other'].append(param)

    param_list = [{'params': no_decay[k], 'weight_decay': 0., 'lr': lr[k]} for k in lr.keys()] + \
                 [{'params': decay[k], 'weight_decay': weight_decay, 'lr': lr[k]} for k in lr.keys()]
    return param_list


class Dist_KLD(torch.nn.Module):

    def __init__(self, dist_temp):
        super(Dist_KLD, self).__init__()
        self.dist_temp = dist_temp

    def forward(self, student_logits, teacher_gt, teacher_dist):
        prob_fusion_teacher = (F.softmax(teacher_gt/self.dist_temp, dim=-1) + F.softmax(teacher_dist/self.dist_temp, dim=-1))/2
        student_logits = student_logits/self.dist_temp
        loss = (F.softmax(student_logits, dim=-1)*(F.log_softmax(student_logits, dim=-1) - torch.log(prob_fusion_teacher))).sum(-1).mean() * (self.dist_temp * self.dist_temp)
        return loss

