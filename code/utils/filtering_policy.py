import os
import os.path as osp
import numpy as np
import torch


def filtering_points(args, dataset_path: str, epoch: int, ):
    assert osp.exists(dataset_path), f'path {dataset_path} does not exist'
    scenes = os.listdir(osp.join(dataset_path, 'scans'))
    for scene in scenes:
        scene_loss_path = osp.join(dataset_path, 'scans', scene, f'{scene}_loss_iter_{str(epoch)}.npy')
        filtered_point_indices = np.argpartition(np.load(scene_loss_path), args.update_points_num-1)[:args.update_point_num]
        

    return