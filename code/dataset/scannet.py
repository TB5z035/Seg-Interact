import logging
import os
import os.path as osp
import re
from collections import namedtuple

import MinkowskiEngine as ME
import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset

from . import register_dataset
from .transforms import TRANSFORMS, Compose, parse_transform

is_valid_scene_id = re.compile(r'^scene\d{4}_\d{2}$').match
logger = logging.getLogger('scannet')


def read_plyfile(path, label_path=None):
    """
    Read a .ply file and return a tuple of (vertices, faces, colors, normals, faces_colors, faces_normals)

    Example:

    PlyData(
        (PlyElement('vertex',
                    (PlyProperty('x', 'float'), PlyProperty('y', 'float'),
                     PlyProperty('z', 'float'), PlyProperty('red', 'uchar'),
                     PlyProperty('green', 'uchar'), PlyProperty('blue', 'uchar'),
                     PlyProperty('alpha', 'uchar')),
                    count=80583,
                    comments=[]),
         PlyElement('face',
                    (PlyListProperty('vertex_indices', 'uchar', 'int'), ),
                    count=151828,
                    comments=[])),
        text=False,
        byte_order='<',
        comments=['VCGLIB generated'],
        obj_info=[])

    PlyData((PlyElement(
        'vertex',
        (PlyProperty('x', 'float'), PlyProperty('y', 'float'),
         PlyProperty('z', 'float'), PlyProperty('red', 'uchar'),
         PlyProperty('green', 'uchar'), PlyProperty('blue', 'uchar'),
         PlyProperty('alpha', 'uchar'), PlyProperty('label', 'ushort')),
        count=80583,
        comments=[]),
             PlyElement('face',
                        (PlyListProperty('vertex_indices', 'uchar', 'int'), ),
                        count=151828,
                        comments=[])),
            text=False,
            byte_order='<',
            comments=['MLIB generated'],
            obj_info=[])

    Returns:
        coords: (N, 3) float32
        colors: (N, 4) uint8
        faces: (M, 3) int32
        labels: (N,) uint16 or None

    """
    plydata = PlyData.read(path)
    coords = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=1)
    colors = np.stack(
        [plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue'], plydata['vertex']['alpha']],
        axis=1)
    faces = np.stack(plydata['face']['vertex_indices'], axis=0)
    if label_path is not None:
        label_plydata = PlyData.read(label_path)
        labels = np.array(label_plydata['vertex']['label'])
    else:
        labels = None

    return coords, colors, faces, labels


Label = namedtuple('Label', ['id', 'train_id', 'name', 'color'])


@register_dataset('scannet')
class ScannetDataset(Dataset):
    """
    Scannet dataset

    The structure of ScanNet Dataset:

    DATASET_ROOT
    ├── ext_info
    │   ├── test.txt
    │   ├── train.txt
    │   └── val.txt
    ├── scannetv2-labels.combined.tsv
    ├── scans 
    │   ├── scene0000_00 
    │   │   ├── scene0000_00_vh_clean_2.0.010000.segs.json
    │   │   ├── scene0000_00_vh_clean_2.labels.ply *
    │   │   ├── scene0000_00_vh_clean_2.ply *
    │   │   ├── scene0000_00_vh_clean.aggregation.json
    │   │   └── scene0000_00_vh_clean.ply
    │   ├── scene0000_01
    │   ├── scene0000_02
    │   ├── scene0001_00
    │   ├── ...
    │   └── scene0706_00
    └── scans_test
        ├── scene0707_00
        ├── scene0708_00
        ├── ...
        └── scene0806_00

    *: must have
    
    scene_id: e.g. scene0000_00
    """

    LABEL_PROTOCOL = [
        Label(0, 255, "unannotated", (0, 0, 0)),
        Label(1, 0, "wall", (174, 199, 232)),
        Label(2, 1, "floor", (152, 223, 138)),
        Label(3, 2, "cabinet", (31, 119, 180)),
        Label(4, 3, "bed", (255, 187, 120)),
        Label(5, 4, "chair", (188, 189, 34)),
        Label(6, 5, "sofa", (140, 86, 75)),
        Label(7, 6, "table", (255, 152, 150)),
        Label(8, 7, "door", (214, 39, 40)),
        Label(9, 8, "window", (197, 176, 213)),
        Label(10, 9, "bookshelf", (148, 103, 189)),
        Label(11, 10, "picture", (196, 156, 148)),
        Label(12, 11, "counter", (23, 190, 207)),
        Label(13, 255, "blinds", (178, 76, 76)),
        Label(14, 12, "desk", (247, 182, 210)),
        Label(15, 255, "shelves", (66, 188, 102)),
        Label(16, 13, "curtain", (219, 219, 141)),
        Label(17, 255, "dresser", (140, 57, 197)),
        Label(18, 255, "pillow", (202, 185, 52)),
        Label(19, 255, "mirror", (51, 176, 203)),
        Label(20, 255, "floormat", (200, 54, 131)),
        Label(21, 255, "clothes", (92, 193, 61)),
        Label(22, 255, "ceiling", (78, 71, 183)),
        Label(23, 255, "books", (172, 114, 82)),
        Label(24, 14, "refridgerator", (255, 127, 14)),
        Label(25, 255, "television", (91, 163, 138)),
        Label(26, 255, "paper", (153, 98, 156)),
        Label(27, 255, "towel", (140, 153, 101)),
        Label(28, 15, "shower curtain", (158, 218, 229)),
        Label(29, 255, "box", (100, 125, 154)),
        Label(30, 255, "whiteboard", (178, 127, 135)),
        Label(31, 255, "person", (120, 185, 128)),
        Label(32, 255, "nightstand", (146, 111, 194)),
        Label(33, 16, "toilet", (44, 160, 44)),
        Label(34, 17, "sink", (112, 128, 144)),
        Label(35, 255, "lamp", (96, 207, 209)),
        Label(36, 18, "bathtub", (227, 119, 194)),
        Label(37, 255, "bag", (213, 92, 176)),
        Label(38, 255, "otherstructure", (94, 106, 211)),
        Label(39, 19, "otherfurniture", (82, 84, 163)),
        Label(40, 255, "otherprop", (100, 85, 144)),
    ]

    @property
    def train_class_names(self):
        return [l.name for l in self.LABEL_PROTOCOL if l.train_id != 255]

    @property
    def num_train_classes(self):
        return len([l for l in self.LABEL_PROTOCOL if l.train_id != 255])

    @property
    def ignore_class(self):
        return 255

    @property
    def num_channel(self):
        return 3

    SPLIT_PATHS = {
        'train': 'scans',
        'val': 'scans',
        'test': 'scans_test',
    }

    def _collate_fn(self, batch):
        inputs, labels, extras = list(zip(*batch))
        coords, faces, feats = list(zip(*inputs))
        indices = torch.cat([torch.ones_like(c[..., :1]) * i for i, c in enumerate(coords)], 0)
        bcoords = torch.cat((indices, *coords), -1)
        bfeats = torch.cat(feats, 0)
        bfaces = torch.cat(faces, 0)
        blabels = torch.cat(labels, 0)
        bextras = {}
        for key in extras[0].keys():
            bextras[key] = tuple([extra[key] for extra in extras])
        return bcoords, bfeats, bfaces, blabels, bextras

    def __init__(self, root, split='train', transform=[]):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = parse_transform(transform)
        self._load_scene_ids(split)

    def _load_scene_ids(self, split):
        """
        Load scene ids from a relative path
        the relative path is in SPLIT_PATH

        Set self.scene_ids
        
        Returns:
            None
        """
        with open(osp.join(self.root, 'ext_info', f'{split}.txt'), 'r') as f:
            self.scene_ids = [i.strip() for i in f.readlines()]
        # Cross check with the directory
        rel_path = self.SPLIT_PATHS[split]
        scene_ids_list = [i for i in os.listdir(osp.join(self.root, rel_path)) if is_valid_scene_id(i)]
        for i in self.scene_ids:
            assert i in scene_ids_list, f'{i} not found in {rel_path}'

    def _load_ply(self, scene_path):
        """
        Load a scene from a scene directory path

        scene_path: path to a scene. root / split_path / scene_id

        Returns:
            coords: (N, 3) float32
            colors: (N, 4) uint8
            faces: (M, 3) int32
            labels: (N,) uint16 or None
        """
        data_suffix = '_vh_clean_2.ply'
        label_suffix = '_vh_clean_2.labels.ply'
        data_ply = [i for i in os.listdir(scene_path) if i.endswith(data_suffix)]
        label_ply = [i for i in os.listdir(scene_path) if i.endswith(label_suffix)]
        assert len(data_ply) == 1, f'Found {len(data_ply)} data ply files in {scene_path}'

        data_path = osp.join(scene_path, data_ply[0])
        label_path = osp.join(scene_path, label_ply[0]) if len(label_ply) == 1 else None

        coords, colors, faces, labels = read_plyfile(data_path, label_path)
        return coords, colors, faces, self._convert_labels(labels)

    def _convert_labels(self, labels=None):
        """
        Convert labels to train ids

        labels: numpy.ndarray (N,) uint16 or None

        Returns:
            labels: (N,) uint16
        """
        if labels is None:
            return None
        train_labels = np.ones_like(labels) * 255
        for l in self.LABEL_PROTOCOL:
            train_labels[labels == l.id] = l.train_id
        return train_labels

    def label_trainid_2_id(self, train_ids=None):
        """
        Convert train ids to ids

        labels: numpy.ndarray (N,) uint16 or None

        Returns:
            labels: (N,) uint16
        """
        if train_ids is None:
            return None
        train_labels = np.ones_like(train_ids) * 255
        for l in self.LABEL_PROTOCOL:
            if l.train_id != 255:
                train_labels[train_ids == l.train_id] = l.id
        return train_labels

    # id2path = lambda self, scene_id: osp.join(self.root, self.SPLIT_PATHS[self.split], scene_id)

    def _prepare_item(self, index):
        """
        Get a scene from the dataset
        """
        scene_path = osp.join(self.root, self.SPLIT_PATHS[self.split], self.scene_ids[index])
        coords, colors, faces, labels = self._load_ply(scene_path)

        (coords, faces, colors), labels, extra = self.transform((coords, faces, colors[:, :3]), labels, {
            'scene_path': scene_path,
            'scene_id': self.scene_ids[index],
            'gt_labels': np.copy(labels),
        })
        return (coords, faces, colors), labels, extra

    def __getitem__(self, index):
        """
        extra keys: scene_path, scene_id, gt_labels
        """
        (coords, faces, colors), labels, extra = self._prepare_item(index)
        coords = torch.from_numpy(coords)
        colors = torch.from_numpy(colors)
        faces = torch.from_numpy(faces)
        labels = torch.from_numpy(labels.astype(np.int64))
        return (coords, faces, colors), labels, extra

    def __len__(self):
        return len(self.scene_ids)


@register_dataset('scannet_quantized')
class ScanNetQuantized(ScannetDataset):
    VOXEL_SIZE = 0.02

    def _collate_fn(self, batch):
        """
        extra keys: scene_path, scene_id, gt_labels, maps
        """
        inputs, labels, extras = list(zip(*batch))
        coords, faces, feats = list(zip(*inputs))
        indices = torch.cat([torch.ones_like(c[..., :1]) * i for i, c in enumerate(coords)], 0)
        bcoords = torch.cat((indices, torch.cat(coords, 0)), -1)
        bfeats = torch.cat(feats, 0)
        bfaces = torch.cat(faces, 0)
        blabels = torch.cat(labels, 0)
        bextras = {}
        for key in extras[0].keys():
            bextras[key] = tuple([extra[key] for extra in extras])

        ## Collate maps and inverse maps
        maps = bextras['maps']
        map_list, inv_map_list = list(zip(*maps))
        map_cum_length = torch.cumsum(torch.tensor([0] + [len(m) for m in inv_map_list]), 0)
        map_indices = torch.cat([torch.ones_like(c) * i for i, c in enumerate(map_list)], 0)
        map_bias = map_cum_length[map_indices.to(int)]
        bmap = torch.cat(map_list, 0) + map_bias

        inv_map_cum_length = torch.cumsum(torch.tensor([0] + [len(m) for m in map_list]), 0)
        inv_map_indices = torch.cat([torch.ones_like(c) * i for i, c in enumerate(inv_map_list)], 0)
        inv_map_bias = inv_map_cum_length[inv_map_indices.to(int)]
        binv_map = torch.cat(inv_map_list, 0) + inv_map_bias

        return (bcoords, bfaces, bfeats), blabels, bextras | {'maps': (bmap, binv_map), 'batch_indices': indices}

    # (bmap, binv_map)

    def __getitem__(self, index) -> dict:
        """
        extra keys: scene_path, scene_id, gt_labels, maps
        """
        (coords, faces, colors), labels, extra = self._prepare_item(index)

        coords = torch.from_numpy(coords)
        colors = torch.from_numpy(colors)
        faces = torch.from_numpy(faces)
        labels = torch.from_numpy(labels.astype(np.int64))

        coords = (coords / self.VOXEL_SIZE).to(torch.int32)
        unique_map, inverse_map = ME.utils.quantization.unique_coordinate_map(coords)
        coords = coords[unique_map].to(torch.float64)
        colors = colors[unique_map]
        labels = labels[unique_map]

        return (coords, faces, colors), labels, extra | {'maps': (unique_map, inverse_map)}


@register_dataset('scannet_quantized_limited')
class ScanNetQuantizedLimited(ScanNetQuantized):

    def __init__(self, root, split='train', transform=None, limit=None):
        super().__init__(root, split, transform)
        assert limit in [20, 50, 100, 200], f'Invalid limit {limit}'
        assert osp.exists(osp.join(
            root, 'data_efficient')), 'Data efficient specifications ($(ROOT)/data_efficient) not found'
        self.limit = limit
        self.limit_dict = torch.load(osp.join(root, 'data_efficient', 'points', f'points{limit}'))

    def _prepare_item(self, index):
        scene_path = osp.join(self.root, self.SPLIT_PATHS[self.split], self.scene_ids[index])
        coords, colors, faces, labels = self._load_ply(scene_path)
        original_labels = np.copy(labels)

        # Use limited labels
        scene_id = self.scene_ids[index]
        limit = self.limit_dict[scene_id]
        mask = np.ones_like(labels, dtype=bool)
        mask[limit] = False
        labels[mask] = 255

        (coords, faces, colors), labels, extra = self.transform((coords, faces, colors[:, :3]), labels, {
            'scene_path': scene_path,
            'scene_id': self.scene_ids[index],
            'gt_labels': original_labels,
        })
        return (coords, faces, colors), labels, extra


class FastLoad(ScannetDataset):
    _round = None
    _label_root = None

    @property
    def label_root(self):
        return self._label_root

    @label_root.setter
    def label_root(self, label_root):
        self._label_root = label_root

    @property
    def round(self):
        return self._round

    @round.setter
    def round(self, round):
        self._round = round

    @property
    def round_suffix(self):
        return f'_{self.round}' if self.round is not None else ''

    def _load_ply(self, scene_path):
        fast_scene_files = [i for i in os.listdir(scene_path) if i.endswith('_scene.npy')]
        if len(fast_scene_files) < 1:
            logger.warning(f'FastLoad fails: No processed npy files found in {scene_path}')
            return super()._load_ply(scene_path)
        assert len(fast_scene_files) == 1, f'Found {len(fast_scene_files)} scene files in {scene_path}'
        scene_id = re.findall(r'(.*)_scene.npy', fast_scene_files[0])[0]

        scene_obj = np.load(os.path.join(scene_path, f'{scene_id}_scene.npy'), allow_pickle=True)[None][0]
        coords = scene_obj['coords']
        colors = scene_obj['colors']
        faces = scene_obj['faces']

        label_root = osp.join(self.label_root, scene_id) if self.label_root else scene_path
        if len([i for i in os.listdir(label_root) if re.findall(r'(.*)_labels(.*).npy', i)]) > 0:
            fast_label_files = [i for i in os.listdir(label_root) if i.endswith(f'_labels{self.round_suffix}.npy')]
            assert len(fast_label_files
                      ) == 1, f'Found {len(fast_label_files)} label files at round {self.round} in {label_root}'
            labels = np.load(os.path.join(label_root, f'{scene_id}_labels{self.round_suffix}.npy'),
                             allow_pickle=True)[None][0]
        else:
            labels = None
        return coords, colors, faces, self._convert_labels(labels)

    def save_pred(self, scene_path: str, arr: np.ndarray, suffix: str = 'labels', save_root=None):
        scene_id = re.findall(r'scene\d{4}\_\d{2}', scene_path)[0]
        save_root = osp.join(save_root, scene_id) if save_root is not None else scene_path
        os.makedirs(save_root, exist_ok=True)
        np.save(os.path.join(save_root, f'{scene_id}_{suffix}{self.round_suffix}.npy'), arr)


@register_dataset('scannet_quantized_fast')
class ScanNetQuantizedFast(ScanNetQuantized, FastLoad):
    pass


@register_dataset('scannet_quantized_limited_fast')
class ScanNetQuantizedLimitedFast(ScanNetQuantizedLimited, FastLoad):
    pass
