import os
import os.path as osp
import numpy as np
import logging
from tqdm import tqdm

from . import register_vis_type

logger = logging.getLogger('Visualization')


def clean_txt_files(point_cloud_path: str):
    scenes = sorted(os.listdir(point_cloud_path))
    for scene in scenes:
        del_files = [i for i in os.listdir(osp.join(point_cloud_path, scene)) if i.endswith('.txt')]
        for file in del_files:
            os.remove(osp.join(point_cloud_path, scene, file))


def gen_excluded_points(point_cloud_path: str, scene: str, output_excluded=False):
    if osp.exists(osp.join(point_cloud_path, scene, f'{scene}_updated_indices.npy')):
        scene_data = np.load(osp.join(point_cloud_path, scene, f'{scene}_base_coords_colors.npy'), allow_pickle=True)
        filtered_indices = np.load(osp.join(point_cloud_path, scene, f'{scene}_updated_indices.npy'), allow_pickle=True)
        excluded_indices = np.delete(np.arange(len(scene_data)), filtered_indices)
        excluded_points = scene_data[excluded_indices]
        if not osp.exists(osp.join(point_cloud_path, scene, f'{scene}_excluded_coords_colors.npy')):
            np.save(osp.join(point_cloud_path, scene, f'{scene}_excluded_coords_colors.npy'),
                    excluded_points,
                    allow_pickle=True)
        if output_excluded:
            return excluded_indices


def set_color(coords_colors: np.ndarray, chosen_color: str):
    color_dict = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'pink': (255, 85, 255),
        'orange': (255, 85, 0),
        'yellow': (255, 255, 0),
        'blue': (0, 0, 255),
        'cyan': (0, 255, 255),
        'green': (0, 255, 0),
        'purple': (170, 0, 255)
    }
    try:
        r, g, b = color_dict[chosen_color]
    except:
        raise ValueError(
            f'chosen color: {chosen_color} is not in the color library, please choose from:\n(white, black, red, pink, orange, yellow, blue, cyan, green, purple)'
        )
    coords_colors[:, 3] = r
    coords_colors[:, 4] = g
    coords_colors[:, 5] = b
    return coords_colors


class vis_base():

    def __init__(self, dataset: object, inference_save_path: str, point_cloud_path: str):
        self.dataset = dataset
        self.dataset_path = dataset.root
        self.point_cloud_path = point_cloud_path
        self.inference_save_path = inference_save_path
        assert osp.exists(self.dataset_path), 'path to scannet dataset does not exist'
        assert osp.exists(self.point_cloud_path), 'path to visualization files does not exist'
        assert osp.exists(self.inference_save_path), 'path to saved inference files does not exist'


@register_vis_type('highlight_updated')
class highlight_updated(vis_base):

    def __call__(self):
        logger.info("Visualizing with highlight_updated")
        clean_txt_files(self.point_cloud_path)
        scenes = sorted(os.listdir(self.point_cloud_path))
        color_sequence = ('white', 'red', 'pink', 'orange', 'yellow', 'blue', 'cyan', 'green')
        for scene in tqdm(scenes):
            gen_excluded_points(self.point_cloud_path, scene)

            # Overrides the Colors of the Updated Labels for Clarity
            color_files = sorted([
                i for i in os.listdir(osp.join(self.point_cloud_path, scene))
                if not (i.endswith('_updated_indices.npy') or i.endswith('_coords_colors.npy')) and i.endswith('.npy')
            ])
            if len(color_files) != 0:
                all_coords_colors = np.zeros((1, 7))
                for file in color_files:
                    file_color = color_sequence[color_files.index(file) % len(color_sequence)]
                    coords_colors = np.load(osp.join(self.point_cloud_path, scene, file))
                    coords_colors = set_color(coords_colors, file_color)
                    all_coords_colors = np.concatenate((all_coords_colors, coords_colors), axis=0)
                np.save(osp.join(self.point_cloud_path, scene, f'{scene}_highlight_updated_coords_colors.npy'),
                        all_coords_colors[1:],
                        allow_pickle=True)


@register_vis_type('color_by_segment')
class color_by_segment(vis_base):

    def __call__(self):
        logger.info("Visualizing with color_by_segment")
        clean_txt_files(self.point_cloud_path)
        scenes = sorted(os.listdir(self.inference_save_path))
        for scene in tqdm(scenes):
            # Sets Segmentation Color for Excluded Points
            pred_labels, gt_labels = np.load(osp.join(self.inference_save_path, scene,
                                                      f'{scene}_labels_iter_final.npy'),
                                             allow_pickle=True)
            datapoints = np.load(osp.join(self.point_cloud_path, scene, f'{scene}_base_coords_colors.npy'),
                                 allow_pickle=True)
            pred_colors = self.dataset.find_match_color(pred_labels)
            gt_colors = self.dataset.find_match_color(gt_labels)
            datapoints[:, 3:] = pred_colors[:]
            np.save(osp.join(self.point_cloud_path, scene, f'{scene}_pred_seg_coords_colors.npy'),
                    datapoints,
                    allow_pickle=True)
            datapoints[:, 3:] = gt_colors[:]
            np.save(osp.join(self.point_cloud_path, scene, f'{scene}_gt_seg_coords_colors.npy'),
                    datapoints,
                    allow_pickle=True)


@register_vis_type('color_by_preds')
class color_by_preds(vis_base):

    def __call__(self):
        logger.info("Visualizing with color_by_preds")
        clean_txt_files(self.point_cloud_path)
        scenes = sorted(os.listdir(self.inference_save_path))
        for scene in tqdm(scenes):
            # Sets Colors for Correctly and Incorrectly Predicted Points
            pred_labels, gt_labels = np.load(osp.join(self.inference_save_path, scene,
                                                      f'{scene}_labels_iter_final.npy'),
                                             allow_pickle=True)
            datapoints = np.load(osp.join(self.point_cloud_path, scene, f'{scene}_base_coords_colors.npy'),
                                 allow_pickle=True)
            error_indices = np.where(pred_labels != gt_labels)
            error_preds = datapoints[error_indices]
            error_pred_coords_colors = set_color(error_preds, 'white')
            correct_pred_coords_colors = np.delete(datapoints, error_indices, axis=0)
            np.save(osp.join(self.point_cloud_path, scene, f'{scene}_correct_coords_colors.npy'),
                    correct_pred_coords_colors,
                    allow_pickle=True)
            np.save(osp.join(self.point_cloud_path, scene, f'{scene}_error_coords_colors.npy'),
                    error_pred_coords_colors,
                    allow_pickle=True)
