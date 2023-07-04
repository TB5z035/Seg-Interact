import logging
import os
import os.path as osp
import time

import torch


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(osp.join(args.exp_dir, 'logs', f'{get_time_str()}.txt'), mode='w')
    ch = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[fh, ch])
