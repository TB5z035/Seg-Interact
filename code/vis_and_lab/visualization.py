# import open3d as o3d
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

from ..utils.misc import gen_excluded_pts_file, npy_to_txt, set_color


def prep_files_for_visuaization(point_cloud_path: str):
    gen_excluded_pts_file(point_cloud_path)

    scenes = sorted(os.listdir(point_cloud_path))
    color_sequence = ('white', 'red', 'pink', 'orange', 'yellow', 'blue', 'cyan', 'green')
    for scene in tqdm(scenes):
        color_files = sorted([
            i for i in os.listdir(osp.join(point_cloud_path, scene))
            if not (i.endswith('_updated_indices.npy') or i.endswith('_coords_colors.npy')) and i.endswith('.npy')
        ])
        for file in color_files:
            file_color = color_sequence[color_files.index(file) % len(color_sequence)]
            coords_colors = np.load(osp.join(point_cloud_path, scene, file))
            coords_colors = set_color(coords_colors, file_color)
            np.save(osp.join(point_cloud_path, scene, file), coords_colors, allow_pickle=True)

    npy_to_txt(point_cloud_path, point_cloud_path)


def vis_point_cloud(point_cloud_path: str):
    pass


def highlight_updated(point_cloud_path: str):
    pass
