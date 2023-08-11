import os
import os.path as osp
import numpy as np
from tqdm import tqdm


def save_vis_file(dataset_path: str, scene_id: str, save_path: str):
    '''
    input file: _scene.npy
                np.ndarray({'coords': [...],
                            'colors': [...],
                            'faces': [...]})
    output file: _coords_colors.npy
                 [x1, y1, z1, r1, g1, b1, a1]
                 ...
                 [xn, yn, zn, rn, gn, bn, an]
    '''
    scene_path = osp.join(dataset_path, 'scans', scene_id)
    if osp.exists(scene_path):
        scene_data = np.load(osp.join(scene_path, f'{scene_id}_scene.npy'), allow_pickle=True)[None][0]
        coords, colors = scene_data['coords'], scene_data['colors']
        # colors = np.divide(colors, 255)
        assert len(coords) == len(colors), f'coords len {len(coords)} and color len {len(colors)} mismatch'
        #alpha = colors[:, 3]
        #alpha = np.expand_dims(np.floor_divide(alpha, 2), axis=1)
        #xyzrgba = np.concatenate((coords, colors[:, :3], alpha), axis=1)
        xyzrgba = np.concatenate((coords, colors), axis=1)
        assert osp.exists(save_path), f'save path {save_path} does not exist'
        os.makedirs(osp.join(save_path, scene_id), exist_ok=True)
        np.save(osp.join(save_path, scene_id, f'{scene_id}_base_coords_colors.npy'), xyzrgba)


def vis_preprocess(dataset_path: str, save_path: str):
    scenes = sorted(os.listdir(osp.join(dataset_path, 'scans')))
    for scene in tqdm(scenes):
        save_vis_file(dataset_path, scene, save_path)


if __name__ == '__main__':
    dataset_path = '/home/Guest/tb5zhh/datasets/ScanNet'
    save_path = '/home/Guest/caiz/labeling_inference/visualize/scannet_scenes'
    vis_preprocess(dataset_path, save_path)
