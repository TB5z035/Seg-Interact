import os
import os.path as osp
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger('labeling_inference: filtering')


def highest_loss_filtering(args, inf_save_path: str, epoch: int):
    logger.info(f"Filtering points from with highest loss criterions {epoch}")
    assert osp.exists(inf_save_path), f'dataset path {inf_save_path} does not exist'
    scenes = os.listdir(inf_save_path)
    print(len(scenes))
    for scene in tqdm(scenes):
        scene_loss_path = osp.join(inf_save_path, scene, f'{scene}_loss_iter_{epoch}.npy')
        scene_labels_path = osp.join(inf_save_path, scene, f'{scene}_labels_iter_{epoch}.npy')
        prev_updated_path = osp.join(inf_save_path, scene, f'{scene}_updated_labels_iter_{epoch-1}.npy')
        if osp.exists(scene_loss_path) and osp.exists(scene_labels_path) and osp.exists(prev_updated_path):
            scene_losses, prev_updated_labels, (
                scene_predictions,
                scene_gt_labels) = np.load(scene_loss_path), np.load(prev_updated_path), np.load(scene_labels_path)
            assert len(scene_gt_labels) == len(scene_predictions) == len(
                scene_losses), f'number of preds, gt_labels, and losses differ in scene {scene}'

            # Get Existing Label Indices
            existing_label_indices = np.delete(np.arange(len(prev_updated_labels)), np.where(prev_updated_labels == 0))
            existing_num = len(existing_label_indices)

            # Exclude Labeled points in ScanNetLimited Dataset from Selection
            mask = np.zeros_like(scene_losses, dtype=bool)
            mask[existing_label_indices] = True
            scene_losses[mask] = 0.

            # Selection and Update
            update_num = args.update_points_num if (len(scene_losses) -
                                                    existing_num) >= args.update_points_num else (len(scene_losses) -
                                                                                                  existing_num)
            if update_num > 0:
                filtered_point_indices = np.argpartition(scene_losses, -update_num)[-update_num:]
                excluded_point_indices = np.delete(np.arange(len(scene_losses)), filtered_point_indices)
                prev_updated_labels[filtered_point_indices] = scene_gt_labels[filtered_point_indices]
                prev_updated_labels[excluded_point_indices] = 0
                scene_updated_labels = prev_updated_labels
                np.save(osp.join(inf_save_path, scene, f'{scene}_updated_labels_iter_{epoch}.npy'),
                        scene_updated_labels)
            elif update_num == 0:
                np.save(osp.join(inf_save_path, scene, f'{scene}_updated_labels_iter_{epoch}.npy'), prev_updated_labels)
