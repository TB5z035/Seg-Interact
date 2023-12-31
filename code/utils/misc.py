import logging
import os
import os.path as osp
import time

import torch
import torch.distributed as dist

def to_device(data, device):
    if isinstance(data, list) or isinstance(data, tuple):
        return [d.to(device) if d is not None else None for d in data]
    else:
        return data.to(device)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def get_local_rank():
    try:
        return dist.get_rank()
    except:
        return 0
    
def get_world_size():
    try:
        return dist.get_world_size()
    except:
        return 1

def get_device():
    # TODO add support for multi-gpu
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_directory(args, args_text):
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(osp.join(args.exp_dir, 'logs'), exist_ok=True)
    os.makedirs(osp.join(args.exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(osp.join(args.exp_dir, 'visuals'), exist_ok=True)
    os.makedirs(osp.join(args.exp_dir, 'tensorboard'), exist_ok=True)
    with open(osp.join(args.exp_dir, 'config.yaml'), 'w') as f:
        f.write(args_text)



def init_logger(args):
    formatter = logging.Formatter('[%(name)-20s][%(module)-10s L%(lineno)-3d][%(levelname)-8s] %(asctime)s %(msecs)03d:  %(message)s')
    fh = logging.FileHandler(osp.join(args.exp_dir, 'logs', f'{get_time_str()}.txt'), mode='w')
    ch = logging.StreamHandler()
    if get_local_rank() == 0:
        fh.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    else:
        fh.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if get_local_rank() == 0:
        logging.basicConfig(level=logging.INFO, handlers=[fh, ch])
    else:
        logging.basicConfig(level=logging.WARNING, handlers=[ch])
