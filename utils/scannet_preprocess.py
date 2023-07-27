"""
This script extract the point-wise label from ply file with '_vh_clean_2.labels.ply' as suffix.
Labels are stored in numpy array with shape (N, 1), where N is the number of points.
"""
import os
import re

from plyfile import PlyData
import numpy as np
from tqdm import tqdm


def extract_and_save_labels(scene_path: str, root_path=None, save_root_path=None) -> None:
    """
    Extract the labels from a scene specified by scene_path, and save to {scene_id}_labels.npy in the same folder.
    """

    target_save_path = scene_path if save_root_path is None else scene_path.replace(root_path, save_root_path)
    if [i for i in target_save_path if i.endswith('_labels.npy')]:
        print(f'{scene_path} already contains processed labels, skipped')
        return
    assert os.path.exists(scene_path), f'{scene_path} does not exist'
    filtered_list = [i for i in os.listdir(scene_path) if i.endswith('_vh_clean_2.labels.ply')]
    assert filtered_list, f'{scene_path} does not contain any label file'
    assert len([filtered_list]) == 1, f'{scene_path} contains multiple label files'
    label_filename = filtered_list[0]
    scene_id = re.findall(r'(.*)_vh_clean_2.labels.ply', label_filename)[0]
    label_plydata = PlyData.read(os.path.join(scene_path, label_filename))

    if save_root_path is not None:
        os.makedirs(target_save_path, exist_ok=True)
    np.save(os.path.join(target_save_path, f'{scene_id}_labels.npy'), np.array(label_plydata['vertex']['label']))


def extract_and_save_scene(scene_path: str, root_path=None, save_root_path=None) -> None:
    target_save_path = scene_path if save_root_path is None else scene_path.replace(root_path, save_root_path)
    if [i for i in target_save_path if i.endswith('_scene.npy')]:
        print(f'{scene_path} already contains processed scene, skipped')
        return
    assert os.path.exists(scene_path), f'{scene_path} does not exist'
    filtered_list = [i for i in os.listdir(scene_path) if i.endswith('_vh_clean_2.ply')]
    assert filtered_list, f'{scene_path} does not contain any scene file'
    assert len([filtered_list]) == 1, f'{scene_path} contains multiple scene files'
    label_filename = filtered_list[0]
    scene_id = re.findall(r'(.*)_vh_clean_2.ply', label_filename)[0]
    scene_plydata = PlyData.read(os.path.join(scene_path, label_filename))

    if save_root_path is not None:
        os.makedirs(target_save_path, exist_ok=True)
    np.save(
        os.path.join(target_save_path, f'{scene_id}_scene.npy'), {
            'coords':
                np.stack(
                    [
                        scene_plydata['vertex']['x'],
                        scene_plydata['vertex']['y'],
                        scene_plydata['vertex']['z'],
                    ],
                    axis=1,
                ),
            'colors':
                np.stack(
                    [
                        scene_plydata['vertex']['red'],
                        scene_plydata['vertex']['green'],
                        scene_plydata['vertex']['blue'],
                        scene_plydata['vertex']['alpha'],
                    ],
                    axis=1,
                ),
            'faces':
                np.stack(scene_plydata['face']['vertex_indices'], axis=0)
        })


def preprocess_scannet(scannet_path: str, start=0, split=1, save_root=None) -> None:
    """
    Preprocess all the scenes in scannet_path.
    """
    assert os.path.exists(scannet_path), f'{scannet_path} does not exist'
    scenes = [
        root for root, dirs, files in os.walk(scannet_path)
        if [i for i in files if i.endswith('_vh_clean_2.labels.ply')]
    ]

    split_size = (len(scenes) + split - 1) // split
    scenes = scenes[start * split_size:(start + 1) * split_size]
    for scene in tqdm(sorted(scenes)):
        extract_and_save_scene(scene, scannet_path, save_root)
        extract_and_save_labels(scene, scannet_path, save_root)


if __name__ == '__main__':
    import multiprocessing
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess ScanNet dataset')
    parser.add_argument('--scannet_path', '-p', type=str, required=True, help='Path to ScanNet dataset')
    parser.add_argument('--save_root', '-s', type=str, default=None, required=False, help='Path to ScanNet dataset')
    parser.add_argument('--process', '-n', type=int, default=1, help='Number of processes to use')
    args = parser.parse_args()
    pool = multiprocessing.Pool(args.process)
    pool.starmap(preprocess_scannet,
                 [(args.scannet_path, i, args.process, args.save_root) for i in range(args.process)])
