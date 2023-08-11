import os
import os.path as osp
import numpy as np
from tqdm import tqdm

VIS_TYPE = {}


def register_vis_type(type_name):

    def decorator(cls):
        VIS_TYPE[type_name] = cls
        return cls

    return decorator


def npy_to_txt(data_path: str, save_path: str):
    assert osp.exists(data_path), 'data path does not exist'
    assert osp.exists(save_path), f'save path {save_path} does not exist'
    scenes = sorted(os.listdir(data_path))
    for scene in tqdm(scenes):
        scene_files = [
            i for i in os.listdir(osp.join(data_path, scene))
            if not (i.endswith('_base_coords_colors.npy') or i.endswith('_updated_indices.npy')) and
            i.endswith('.npy') and 'iter' not in i
        ]
        for file in scene_files:
            scene_data = np.load(osp.join(data_path, scene, file), allow_pickle=True)
            file_name = file.replace('.npy', '.txt')
            np.savetxt(osp.join(save_path, scene, file_name), scene_data)


def prep_files_for_visuaization(dataset: object, inference_save_path: str, point_cloud_path: str, vis_type: str):
    v = VIS_TYPE[vis_type](dataset, inference_save_path, point_cloud_path)
    v()
    npy_to_txt(point_cloud_path, point_cloud_path)


from . import visualization
