import os
import os.path as osp
import numpy as np
import torch


def highest_loss_filtering(args, dataset_path: str, epoch: int):
    assert osp.exists(dataset_path), f'dataset path {dataset_path} does not exist'
    scenes = os.listdir(osp.join(dataset_path, 'scans'))
    for scene in scenes:
        scene_loss_path = osp.join(dataset_path, 'scans', scene, f'{scene}_loss_iter_{str(epoch)}.npy')
        scene_labels_path = osp.join(dataset_path, 'scans', scene, f'{scene}_labels_iter_{str(epoch)}.npy')
        scene_gt_labels_path = osp.join(dataset_path, 'scans', scene, f'{scene}_labels.npy')
        assert osp.exists(scene_gt_labels_path), f'scene label path {scene_gt_labels_path} does not exist'
        if osp.exists(scene_loss_path) and osp.exists(scene_labels_path):
            scene_losses, scene_predictions, scene_gt_labels = np.load(scene_loss_path), np.load(
                scene_labels_path), np.load(scene_gt_labels_path)

            # Get Existing Label Indices
            if osp.exists(osp.join(dataset_path, 'scans', scene, f'{scene}_updated_labels_iter_{epoch-1}.npy')):
                prev_updated_labels = np.load(
                    osp.join(dataset_path, 'scans', scene, f'{scene}_updated_labels_iter_{epoch-1}.npy'))
                existing_labels = np.delete(np.arange(len(prev_updated_labels)), np.where(prev_updated_labels == -1))
                existing_num = len(existing_labels)
            else:
                existing_num = args.train_dataset["args"]["limit"]
                existing_labels = torch.load(osp.join(dataset_path, 'data_efficient', 'points',
                                                      f'points{existing_num}'))

            assert len(scene_gt_labels) == len(scene_predictions) == len(
                scene_losses), f'number of preds, gt_labels, and losses differ in scene {scene}'
            assert (
                len(scene_losses) - existing_num
            ) >= args.update_point_num, f'exceeded max possible number of points to update: {args.update_point_num} > {len(scene_losses)-existing_num}'

            # Exclude Labeled points in ScanNetLimited Dataset from Selection
            limit = existing_labels[scene]
            mask = np.zeros_like(scene_losses, dtype=bool)
            mask[limit] = True
            scene_losses[mask] = 0.

            # Selection and Update
            filtered_point_indices = np.argpartition(scene_losses, args.update_points_num - 1)[:args.update_point_num]
            excluded_point_indices = np.delete(np.arange(len(scene_losses)), filtered_point_indices)
            scene_predictions[filtered_point_indices] = scene_gt_labels[filtered_point_indices]
            scene_predictions[excluded_point_indices] = -1
            scene_updated_labels = scene_predictions

            np.save(osp.join(dataset_path, 'scans', scene, f'{scene}_updated_labels_iter_{str(epoch)}.npy'),
                    scene_updated_labels)


def random_filtering(args, dataset_path: str, epoch: int):
    assert osp.exists(dataset_path), f'dataset path {dataset_path} does not exist'
    scenes = os.listdir(osp.join(dataset_path, 'scans'))
    for scene in scenes:
        scene_loss_path = osp.join(dataset_path, 'scans', scene, f'{scene}_loss_iter_{str(epoch)}.npy')
        scene_labels_path = osp.join(dataset_path, 'scans', scene, f'{scene}_labels_iter_{str(epoch)}.npy')
        scene_gt_labels_path = osp.join(dataset_path, 'scans', scene, f'{scene}_labels.npy')
        assert osp.exists(scene_gt_labels_path), f'scene label path {scene_gt_labels_path} does not exist'

        if osp.exists(scene_loss_path) and osp.exists(scene_labels_path):
            scene_predictions, scene_gt_labels = np.load(scene_labels_path), np.load(scene_gt_labels_path)

            # Get Existing Label Indices
            if osp.exists(osp.join(dataset_path, 'scans', scene, f'{scene}_updated_labels_iter_{epoch-1}.npy')):
                prev_updated_labels = np.load(
                    osp.join(dataset_path, 'scans', scene, f'{scene}_updated_labels_iter_{epoch-1}.npy'))
                existing_labels = np.delete(np.arange(len(prev_updated_labels)), np.where(prev_updated_labels == -1))
                existing_num = len(existing_labels)
            else:
                existing_num = args.train_dataset["args"]["limit"]
                existing_labels = torch.load(osp.join(dataset_path, 'data_efficient', 'points',
                                                      f'points{existing_num}'))

            assert len(scene_gt_labels) == len(
                scene_predictions), f'number of preds and gt_labels differ in scene {scene}'
            assert (
                len(scene_predictions) - existing_num
            ) >= args.update_point_num, f'exceeded max possible number of points to update: {args.update_point_num} > {len(scene_predictions)-existing_num}'

            # Exclude Labeled points in ScanNetLimited Dataset from Selection
            limit = existing_labels[scene]
            mask = np.zeros_like(scene_predictions, dtype=bool)
            mask[limit] = True
            indices = np.arange(len(scene_predictions))
            indices = np.delete(indices, mask)

            # Selection and Update
            np.random.seed(np.random.randint(0, 50))
            filtered_point_indices = np.random.choice(indices, args.update_point_num, replace=False)
            scene_predictions[filtered_point_indices] = scene_gt_labels[filtered_point_indices]
            scene_updated_labels = scene_predictions
            np.random.seed(args.seed)

            np.save(osp.join(dataset_path, 'scans', scene, f'{scene}_updated_labels_iter_{str(epoch)}.npy'),
                    scene_updated_labels)
