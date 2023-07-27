import os
import os.path as osp
import numpy as np


def highest_loss_filtering(args, dataset_path: str, epoch: int):
    assert osp.exists(dataset_path), f'dataset path {dataset_path} does not exist'
    scenes = os.listdir(osp.join(dataset_path, 'scans'))
    for scene in scenes:
        scene_loss_path = osp.join(dataset_path, 'scans', scene, f'{scene}_loss_iter_{str(epoch)}.npy')
        scene_labels_path = osp.join(dataset_path, 'scans', scene, f'{scene}_labels_iter_{str(epoch)}.npy')
        scene_gt_labels_path = osp.join(dataset_path, 'scans', scene, f'{scene}_labels.npy')
        assert osp.exists(scene_gt_labels_path), f'scene label path {scene_gt_labels_path} does not exist'

        if osp.exists(scene_loss_path) and osp.exists(scene_labels_path):
            scene_losses = np.load(scene_loss_path)
            filtered_point_indices = np.argpartition(scene_losses, args.update_points_num - 1)[:args.update_point_num]
            scene_predictions, scene_gt_labels = np.load(scene_labels_path), np.load(scene_gt_labels_path)
            scene_predictions[filtered_point_indices] = scene_gt_labels[filtered_point_indices]
            scene_updated_labels = scene_predictions

            np.save(osp.join(dataset_path, 'scans', scene, f'{scene}_updated_labels_iter_{str(epoch)}.npy'),
                    scene_updated_labels)
