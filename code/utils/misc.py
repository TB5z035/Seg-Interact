import logging
import os
import os.path as osp
import re
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


def save_pseudo_labels(labels: np.ndarray, save_path: str, scene_id: str, epoch) -> None:
    assert osp.exists(save_path), f'dataset path {save_path} does not exist'
    scene_path = osp.join(save_path, scene_id)
    os.makedirs(scene_path, exist_ok=True)
    np.save(osp.join(scene_path, f'{scene_id}_labels_iter_{epoch}.npy'), labels)


def save_pseudo_loss(loss: np.ndarray, save_path: str, scene_id: str, epoch) -> None:
    assert osp.exists(save_path), f'dataset path {save_path} does not exist'
    scene_path = osp.join(save_path, scene_id)
    os.makedirs(scene_path, exist_ok=True)
    np.save(osp.join(scene_path, f'{scene_id}_loss_iter_{epoch}.npy'), loss)


def clean_inference_paths(save_path: str) -> None:
    '''
    Used to clear pseudo labeling paths
    '''
    scenes = os.listdir(save_path)
    for scene in scenes:
        scene_del_files = [i for i in os.listdir(osp.join(save_path, scene))]
        for file in scene_del_files:
            # print(osp.join(dataset_path, 'scans', scene, file))
            os.remove(osp.join(save_path, scene, file))


def clean_prev_inf_paths(save_path: str) -> None:
    '''
    Used to clear unused pseudo labeling paths
    '''
    scenes = os.listdir(save_path)
    for scene in scenes:
        scene_files = [i for i in os.listdir(osp.join(save_path, scene))]
        suffix = [re.findall(r'[-+]?\d+.npy', f) for f in scene_files]
        epoch_nums = list(map(int, np.concatenate([re.findall(r'[-+]?\d+', f[0]) for f in suffix])))
        epoch_num = np.max(epoch_nums)
        scene_del_files = [i for i in os.listdir(osp.join(save_path, scene)) if not i.endswith(f'iter_{epoch_num}.npy')]
        for file in scene_del_files:
            # print(osp.join(save_path, 'scans', scene, file))
            os.remove(osp.join(save_path, scene, file))


def seq_2_ordered_set(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}")
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {k[len("module") + 1:]: v for k, v in model_state_dict_1.items()}

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {k[len("module") + 1:]: v for k, v in model_state_dict_2.items()}

    for ((k_1, v_1), (k_2, v_2)) in zip(model_state_dict_1.items(), model_state_dict_2.items()):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False


'''Functionalities for Visualizing Point Cloud'''


def clean_vis_paths(vis_path: str):
    assert osp.exists(vis_path), 'path to visualization files does not exist'
    scenes = os.listdir(vis_path)
    for scene in scenes:
        del_files = [i for i in os.listdir(osp.join(vis_path, scene)) if not i.endswith('base_coords_colors.npy')]
        for file in del_files:
            os.remove(osp.join(vis_path, scene, file))
