import logging
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.distributed as dist
import tensorboardX
import yaml


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
    os.makedirs(osp.join(args.exp_dir, 'checkpoints', args.start_time), exist_ok=True)
    os.makedirs(osp.join(args.exp_dir, 'visuals'), exist_ok=True)
    os.makedirs(osp.join(args.exp_dir, 'tensorboard', args.start_time), exist_ok=True)
    os.makedirs(osp.join(args.exp_dir, 'configs'), exist_ok=True)
    with open(osp.join(args.exp_dir, 'configs', f'{args.start_time}.yaml'), 'w') as f:
        f.write(args_text)


def init_logger(args):
    formatter = logging.Formatter(
        '[%(name)-20s][%(module)-10s L%(lineno)-3d][%(levelname)-8s] %(asctime)s %(msecs)03d:  %(message)s')
    fh = logging.FileHandler(osp.join(args.exp_dir, 'logs', f'{args.start_time}.txt'), mode='a')
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

    if get_local_rank() == 0:
        writer = tensorboardX.SummaryWriter(log_dir=osp.join(args.exp_dir, 'tensorboard', args.start_time))
    else:
        writer = None

    return writer


def save_checkpoint(network, args=None, epoch_idx=None, iter_idx=None, optimizer=None, scheduler=None, name='latest'):
    if get_local_rank() == 0:
        torch.save(
            {
                'network': network.module.state_dict(),
                'epoch': epoch_idx if epoch_idx is not None else None,
                'iter': iter_idx if iter_idx is not None else None,
                'optimizer': optimizer.state_dict() if optimizer is not None else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'args': yaml.safe_dump(args.__dict__, default_flow_style=False) if args is not None else None,
            }, f'{args.exp_dir}/checkpoints/{args.start_time}/{name}.pth')


'''Functionalities for Saving Pseudo Label Data'''


def save_pseudo_labels(labels: np.ndarray, dataset_path: str, scene_id: str, epoch) -> None:
    assert osp.exists(dataset_path), f'path {dataset_path} does not exist'
    scene_path = osp.join(dataset_path, 'scans', scene_id)
    # os.makedirs(scene_path, exist_ok=True)
    # print(labels)
    # print(osp.join(scene_path, f'{scene_id}_labels_iter_{str(epoch)}.npy'))
    np.save(osp.join(scene_path, f'{scene_id}_labels_iter_{str(epoch)}.npy'), labels)


def save_pseudo_loss(loss: np.ndarray, dataset_path: str, scene_id: str, epoch) -> None:
    assert osp.exists(dataset_path), f'path {dataset_path} does not exist'
    scene_path = osp.join(dataset_path, 'scans', scene_id)
    # os.makedirs(scene_path, exist_ok=True)
    # print(loss)
    # print(osp.join(scene_path, f'{scene_id}_loss_iter_{str(epoch)}.npy'))
    np.save(osp.join(scene_path, f'{scene_id}_loss_iter_{str(epoch)}.npy'), loss)


def clear_paths(dataset_path: str) -> None:
    '''
    Used to clear pseudo labeling paths (_labels and _loss)
    '''
    scenes = os.listdir(osp.join(dataset_path, 'scans'))
    for scene in scenes:
        path_loss = osp.join(dataset_path, 'scans', scene, f'{scene}_loss_iter_0.npy')
        path_predictions = osp.join(dataset_path, 'scans', scene, f'{scene}_labels_iter_0.npy')
        if osp.exists(path_loss):
            # print(path_loss)
            os.remove(path_loss)
        if osp.exists(path_predictions):
            # print(path_predictions)
            os.remove(path_predictions)
