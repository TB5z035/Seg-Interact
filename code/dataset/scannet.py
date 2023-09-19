import os
import os.path as osp
import re
from collections import namedtuple
import logging

import MinkowskiEngine as ME
import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset

from .transforms import TRANSFORMS, Compose, parse_transform
from . import register_dataset
from .superpoint_base import SuperpointBase
from .scannet_config import *
from ..utils.misc import seq_2_ordered_set
from ..data import Data, Batch, NAG
from .sp_transforms import instantiate_transforms
from .superpoint_base import sp_init
from ..sp_utils import available_cpu_count, starmap_with_kwargs, \
    rodrigues_rotation_matrix, to_float_rgb

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

    def _load_ply_inference(self, new_scene_path, scene_path):
        if not osp.exists(scene_path):
            os.makedirs(scene_path, exist_ok=True)
        inference_scene_files = [i for i in os.listdir(new_scene_path)]
        original_scene_files = [i for i in os.listdir(scene_path)]

        coords, colors, faces, labels = self._load_ply(scene_path)
        if len(inference_scene_files) != 0:
            original_scene_file = [i for i in original_scene_files if i.endswith('_scene.npy')]
            scene_id = re.findall(r'(.*)_scene.npy', original_scene_file[0])[0]
            updated_label_files = [re.findall(r'updated_labels_iter_[-+]?\d+', f) for f in inference_scene_files]
            updated_label_nums = list(
                map(int, np.concatenate([re.findall(r'[-+]?\d+', f[0]) for f in updated_label_files if f != []])))
            iter_num = np.max(updated_label_nums)
            labels = np.load(osp.join(new_scene_path, f'{scene_id}_updated_labels_iter_{iter_num}.npy'))
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
        train_labels = np.zeros_like(train_ids)
        for l in self.LABEL_PROTOCOL:
            if l.train_id != 255:
                train_labels[train_ids == l.train_id] = l.id
        return train_labels

    def find_match_color(self, labels=None):
        """
        finds the corresponding segmentation color
        for a specific label (id)

        labels: numpy.ndarray (N,) uint16 or None

        Return:
            colors: (N,4) uint16 #rgba
        """
        if labels is None:
            return None
        colors = np.zeros((len(labels), 4), dtype=int)
        for l in self.LABEL_PROTOCOL:
            colors[labels == l.id, :] = l.color[0], l.color[1], l.color[2], 255
        return colors

    def _prepare_item(self, index):
        """
        Get a scene from the dataset
        """
        scene_path = osp.join(self.root, self.SPLIT_PATHS[self.split], self.scene_ids[index])
        coords, colors, faces, labels = self._load_ply(scene_path)

        (coords, faces, colors), labels, extra = self.transform((coords, faces, colors[:, :3]), labels, {
            'scene_path': scene_path,
            'scene_id': self.scene_ids[index]
        })
        return (coords, faces, colors), labels, extra

    def __getitem__(self, index):
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
        inputs, labels, extras = list(zip(*batch))
        coords, faces, feats = list(zip(*inputs))
        bextras = {}
        for key in extras[0].keys():
            bextras[key] = tuple(extras[extra_idx]['scene_id']
                                 for extra_idx in range(len(extras))
                                 for _ in range(len(labels[extra_idx]))) if key == 'scene_id' else tuple(
                                     [extra[key] for extra in extras])
        indices = torch.cat([torch.ones_like(c[..., :1]) * i for i, c in enumerate(coords)], 0)
        bcoords = torch.cat((indices, torch.cat(coords, 0)), -1)
        bfeats = torch.cat(feats, 0)
        bfaces = torch.cat(faces, 0)
        blabels = torch.cat(labels, 0)

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

        return (bcoords, bfaces, bfeats), blabels, bextras | {'maps': (bmap, binv_map)}

    def __getitem__(self, index) -> dict:
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

    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 limit=None,
                 labeling_inference=False,
                 inference_save_path=None):
        super().__init__(root, split, transform)
        assert limit in [20, 50, 100, 200], f'Invalid limit {limit}'
        assert osp.exists(osp.join(
            root, 'data_efficient')), 'Data efficient specifications ($(ROOT)/data_efficient) not found'
        self.limit = limit
        self.limit_dict = torch.load(osp.join(root, 'data_efficient', 'points', f'points{limit}'))
        self.labeling_inference = labeling_inference
        self.inf_save_path = inference_save_path

    def _collate_fn(self, batch):
        inputs, labels, extras = list(zip(*batch))
        maps = tuple(extra['maps'] for extra in extras)
        scene_ids = tuple(
            extras[extra_idx]['scene_id'] for extra_idx in range(len(extras)) for _ in range(len(labels[extra_idx])))

        coords, faces, feats = list(zip(*inputs))
        indices = torch.cat([torch.ones_like(c[..., :1]) * i for i, c in enumerate(coords)], 0)
        bcoords = torch.cat((indices, torch.cat(coords, 0)), -1)
        bfeats = torch.cat(feats, 0)
        bfaces = torch.cat(faces, 0)
        blabels = torch.cat(labels, 0)

        map_list, inv_map_list = list(zip(*maps))
        map_cum_length = torch.cumsum(torch.tensor([0] + [len(m) for m in inv_map_list]), 0)
        map_indices = torch.cat([torch.ones_like(c) * i for i, c in enumerate(map_list)], 0)
        map_bias = map_cum_length[map_indices.to(int)]
        bmap = torch.cat(map_list, 0) + map_bias

        inv_map_cum_length = torch.cumsum(torch.tensor([0] + [len(m) for m in map_list]), 0)
        inv_map_indices = torch.cat([torch.ones_like(c) * i for i, c in enumerate(inv_map_list)], 0)
        inv_map_bias = inv_map_cum_length[inv_map_indices.to(int)]
        binv_map = torch.cat(inv_map_list, 0) + inv_map_bias

        return (bcoords, bfaces, bfeats), blabels, {'maps': (bmap, binv_map), 'scene_ids': scene_ids}

    def _prepare_item(self, index):
        scene_path = osp.join(self.root, self.SPLIT_PATHS[self.split], self.scene_ids[index])
        if self.labeling_inference:
            new_scene_path = osp.join(self.inf_save_path, self.scene_ids[index])
            coords, colors, faces, labels = self._load_ply_inference(new_scene_path, scene_path)
        else:
            coords, colors, faces, labels = self._load_ply(scene_path)

        scene_id = self.scene_ids[index]
        limit = self.limit_dict[scene_id]
        mask = np.ones_like(labels, dtype=bool)
        mask[limit] = False
        labels[mask] = 255
        (coords, faces, colors), labels, _ = self.transform((coords, faces, colors[:, :3]), labels, None)
        return (coords, faces, colors), labels, {'path': scene_path}

    def __getitem__(self, index) -> dict:
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

        return (coords, faces, colors), labels, {'maps': (unique_map, inverse_map), 'scene_id': self.scene_ids[index]}


class FastLoad(ScannetDataset):

    def _load_ply(self, scene_path):
        fast_scene_files = [i for i in os.listdir(scene_path) if i.endswith('_scene.npy')]
        fast_label_files = [i for i in os.listdir(scene_path) if i.endswith('_labels.npy')]
        if len(fast_scene_files) < 1:
            logger.warning(f'FastLoad fails: No processed npy files found in {scene_path}')
            return super()._load_ply(scene_path)
        assert len(fast_scene_files) == 1, f'Found {len(fast_scene_files)} scene files in {scene_path}'
        assert len(fast_label_files) == 1, f'Found {len(fast_label_files)} label files in {scene_path}'
        scene_id = re.findall(r'(.*)_scene.npy', fast_scene_files[0])[0]

        scene_obj = np.load(os.path.join(scene_path, f'{scene_id}_scene.npy'), allow_pickle=True)[None][0]
        coords = scene_obj['coords']
        colors = scene_obj['colors']
        faces = scene_obj['faces']

        if f'{scene_id}_labels.npy' in fast_label_files:
            labels = np.load(os.path.join(scene_path, f'{scene_id}_labels.npy'), allow_pickle=True)[None][0]
        else:
            labels = None
        return coords, colors, faces, self._convert_labels(labels)


@register_dataset('scannet_quantized_fast')
class ScanNetQuantizedFast(ScanNetQuantized, FastLoad):
    pass


@register_dataset('scannet_quantized_limited_fast')
class ScanNetQuantizedLimitedFast(ScanNetQuantizedLimited, FastLoad):
    pass


@register_dataset('sp_scannet')
class sp_scannet(SuperpointBase):

    def __init__(self, **kwargs):
        sp_cfg = kwargs['sp_cfg']

        # Superpoint Init
        for key_name in sp_cfg.keys():
            if "transform" in key_name:
                params = getattr(sp_cfg, key_name, None)
                if params is None:
                    continue
                pre_transform = instantiate_transforms(params)

        sp_cfg_dict = sp_cfg.__dict__
        sp_cfg_dict.pop('data_dir', None)
        sp_cfg_dict.pop('pre_transform', None)

        SuperpointBase.__init__(self,
                                root=kwargs['root'],
                                stage=sp_cfg.stage,
                                pre_transform=pre_transform,
                                save_processed_root=sp_cfg.save_processed_root,
                                ignore_label=sp_cfg.ignore_label,
                                **sp_cfg_dict)

    @property
    def dataset_name(self):
        return DATASET_NAME

    @property
    def num_classes(self):
        """Number of classes in the dataset. May be one-item smaller
        than `self.class_names`, to account for the last class name
        being optionally used for 'unlabelled' or 'ignored' classes,
        indicated as `-1` in the dataset labels.
        """
        return ScanNet_NUM_CLASSES

    @property
    def class_names(self):
        """List of string names for dataset classes. This list may be
        one-item larger than `self.num_classes` if the last label
        corresponds to 'unlabelled' or 'ignored' indices, indicated as
        `-1` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.stage)

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...]}`
        """
        scans_paths = [osp.join(self.root, 'scans'), osp.join(self.root, 'scans_test')]
        all_scans = [os.listdir(p) for p in scans_paths]
        return {'scans': all_scans[0], 'scans_test': all_scans[1]}

    def download_dataset(self):
        """
        Check for extracted ScanNet dataset's existence.
        """
        assert len(os.listdir(osp.join(self.root,
                                       'scans'))) != 0, f'No files in ScanNet at {osp.join(self.root, "scans")}'

    @property
    def raw_file_structure(self):
        return f"""
    {self.root}/
        └── scans/
            └── scenexxxx_xx
            └── ...
        └── scans_test/
            └── scenexxxx_xx
            └── ...
            """

    @property
    def raw_file_names(self):
        """The file paths to find in order to skip the download."""
        scan_folders = super().raw_file_names
        return scan_folders

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return self.id_to_base_id(id)

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        return self.read_scannet_scene(raw_cloud_path,
                                       xyz=True,
                                       rgb=True,
                                       semantic=True,
                                       instance=False,
                                       xyz_room=True,
                                       align=False,
                                       is_val=False,
                                       verbose=False)

    def read_scannet_scene(self,
                           scene_dir,
                           xyz=True,
                           rgb=True,
                           semantic=True,
                           instance=False,
                           xyz_room=False,
                           align=False,
                           is_val=False,
                           verbose=False,
                           processes=-1):
        """Read all ScanNet scenes

        :param area_dir: str
            Absolute path to the Area directory, eg: '/some/path/Area_1'
        :param xyz: bool
            Whether XYZ coordinates should be saved in the output Data.pos
        :param rgb: bool
            Whether RGB colors should be saved in the output Data.rgb
        :param semantic: bool
            Whether semantic labels should be saved in the output Data.y
        :param instance: bool
            Whether instance labels should be saved in the output Data.y
        :param xyz_room: bool
            Whether the canonical room coordinates should be saved in the
            output Data.pos_room, as defined in the S3DIS paper section 3.2:
            https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf
        :param align: bool
            Whether the room should be rotated to its canonical orientation,
            as defined in the S3DIS paper section 3.2:
            https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf
        :param is_val: bool
            Whether the output `Batch.is_val` should carry a boolean label
            indicating whether they belong to the Area validation split
        :param verbose: bool
            Verbosity
        :param processes: int
            Number of processes to use when reading rooms. `processes < 1`
            will use all CPUs available
        :return:
            Batch of accumulated points clouds
        """
        # Read all rooms in the Area and concatenate point clouds in a Batch
        assert instance == False, 'does not support instance=True'
        processes = available_cpu_count() if processes < 1 else processes

        coords, colors, _, labels = self._load_ply(scene_path=scene_dir)
        colors = np.delete(colors, 3, axis=1)
        coords, colors, labels = torch.from_numpy(np.float32(coords)), torch.from_numpy(
            np.uint8(colors)), torch.from_numpy(np.int64(labels))
        colors = to_float_rgb(colors)
        batch = Batch.from_data_list([Data(pos=coords, rgb=colors, y=labels, o=None)])

        # Convert from Batch to Data
        data_dict = batch.to_dict()
        del data_dict['batch']
        del data_dict['ptr']
        data = Data(**data_dict)

        return data

    def __len__(self):
        """Number of clouds in the dataset."""
        return len(self.cloud_ids)

    def __getitem__(self, idx):
        """Load a preprocessed NAG from disk and apply `self.transform`
        if any. Optionally, one may pass a tuple (idx, bool) where the
        boolean indicates whether the data should be loaded from disk, if
        `self.in_memory=True`.
        """
        # Prepare from_hdd
        from_hdd = False
        if isinstance(idx, tuple):
            assert len(idx) == 2 and isinstance(idx[1], bool), \
                "Only supports indexing with `int` or `(int, bool)` where the" \
                " boolean indicates whether the data should be loaded from " \
                "disk, when `self.in_memory=True`."
            idx, from_hdd = idx

        # Get the processed NAG directly from RAM
        if self.in_memory and not from_hdd:
            return self.in_memory_data[idx]

        # Read the NAG from HDD
        nag = NAG.load(self.processed_paths[idx], keys_low=self.point_load_keys, keys=self.segment_load_keys)

        # Apply transforms
        nag = nag if self.transform is None else self.transform(nag)

        return nag


@register_dataset('superpoint_scannet')
class superpoint_scannt(FastLoad):

    def __init__(self, sp_cls: object, **kwargs):
        self.sp_cls = sp_cls
        super().__init__(**kwargs)

    def _collate_fn(self, batch):
        inputs, labels, extras = list(zip(*batch))
        coords, colors = list(zip(*inputs))
        bextras = {}
        for key in extras[0].keys():
            bextras[key] = tuple(extras[extra_idx]['scene_id']
                                 for extra_idx in range(len(extras))
                                 for _ in range(len(labels[extra_idx]))) if key == 'scene_id' else tuple(
                                     [extra[key] for extra in extras])

        indices = torch.cat([torch.ones_like(c[..., :1]) * i for i, c in enumerate(coords)], 0)
        bcoords = torch.cat((indices, torch.cat(coords, 0)), -1)
        bcolors = torch.cat(colors, 0)
        blabels = torch.cat(labels, 0)

        return (bcoords, bcolors), blabels, bextras

    def __getitem__(self, index) -> dict:
        scene_id = self.scene_ids[index]
        sp_scene_index = self.sp_cls.cloud_ids.index(scene_id)
        nag = self.sp_cls[sp_scene_index]

        coords, colors, labels = nag[0].pos, nag[0].rgb, nag[0].y.argmax(1)
        linearity, planarity, scattering, elevation = nag[0].linearity, nag[0].planarity, nag[0].scattering, nag[
            0].elevation
        full_super_indices = nag.get_super_index(1, 0)
        superpoint_sizes = nag.get_sub_size(1)

        # Sort data by superpoint indices
        sort = torch.argsort(full_super_indices)
        coords, colors, labels = coords[sort], colors[sort], labels[sort]
        linearity, planarity, scattering, elevation = linearity[sort], planarity[sort], scattering[sort], elevation[
            sort]
        full_super_indices = full_super_indices[sort]
        full_features = torch.concat((coords, colors, linearity, planarity, scattering, elevation), dim=1)

        # Type check
        coords = torch.from_numpy(coords) if type(coords) != torch.Tensor else coords
        colors = torch.from_numpy(colors) if type(colors) != torch.Tensor else colors
        labels = torch.from_numpy(labels.astype(np.int64)) if type(labels) != torch.Tensor else labels
        linearity = torch.from_numpy(linearity) if type(linearity) != torch.Tensor else linearity
        planarity = torch.from_numpy(planarity) if type(planarity) != torch.Tensor else planarity
        scattering = torch.from_numpy(scattering) if type(scattering) != torch.Tensor else scattering
        elevation = torch.from_numpy(elevation) if type(elevation) != torch.Tensor else elevation
        full_features = torch.from_numpy(full_features) if type(full_features) != torch.Tensor else full_features
        full_super_indices = torch.from_numpy(full_super_indices) if type(
            full_super_indices) != torch.Tensor else full_super_indices
        superpoint_sizes = torch.from_numpy(superpoint_sizes) if type(
            superpoint_sizes) != torch.Tensor else superpoint_sizes

        return (coords, colors), labels, {
            'scene_id': scene_id,
            'full_super_indices': full_super_indices,
            'superpoint_sizes': superpoint_sizes,
            'full_features': full_features,
            'linearity': linearity,
            'planarity': planarity,
            'scattering': scattering,
            'elevation': elevation
        }

    def random_masking(self, nag, coords, colors, labels, mask_level=1, ratio=0.6):
        assert int(mask_level) in [1, 2, 3], f'selected masking level: {mask_level} is unsupported'
        coords, colors, labels = coords.numpy(), colors.numpy(), labels.numpy()

        level_num_points = nag.num_points
        # sub_size = nag.get_sub_size(mask_level, 0)
        full_super_indices = nag.get_super_index(mask_level, 0).numpy()
        mask_num = int(round(ratio * level_num_points[mask_level]))
        mask_indices = np.random.choice(np.arange(level_num_points[mask_level]), mask_num, replace=False)
        remain_super_indices = np.array([index for index in full_super_indices if index not in mask_indices])
        mask_super_indices = np.array([index for index in full_super_indices if index in mask_indices])

        rcoords = np.array([coords[i] for i in range(len(coords)) if full_super_indices[i] not in mask_indices])
        rcolors = np.array([colors[i] for i in range(len(colors)) if full_super_indices[i] not in mask_indices])
        rlabels = np.array([labels[i] for i in range(len(labels)) if full_super_indices[i] not in mask_indices])
        mcoords = np.array([coords[i] for i in range(len(coords)) if full_super_indices[i] in mask_indices])
        mcolors = np.array([colors[i] for i in range(len(colors)) if full_super_indices[i] in mask_indices])
        mlabels = np.array([labels[i] for i in range(len(labels)) if full_super_indices[i] in mask_indices])

        return (rcoords, rcolors, rlabels), (mcoords, mcolors, mlabels), (remain_super_indices, mask_super_indices,
                                                                          full_super_indices), mask_level, ratio
