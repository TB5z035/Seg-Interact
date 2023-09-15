# Necessary for advanced config parsing with hydra and omegaconf
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)
import numpy as np
import torch
from .sp_visualization import show
from .dataset.scannet import CLASS_NAMES, CLASS_COLORS
from .dataset.scannet import ScanNet_NUM_CLASSES as NUM_CLASSES
from .dataset.sp_transforms import *
from .utils.args import get_args
from .dataset import DATASETS

cfg, _ = get_args()
cfg_dm = cfg.datamodule

dataset = DATASETS[cfg.datamodule['sp_base']](**(cfg.train_dataset['args'] | {'sp_cfg': cfg_dm}))

nag = dataset[200]

xyz = nag[0].pos
rgb = nag[0].rgb
labels = nag[0].y.argmax(1)

super_indices = nag.get_super_index(3, 0)
indices = torch.where(super_indices == 0)
# print(labels[indices])

show(nag, class_names=CLASS_NAMES, ignore=NUM_CLASSES, class_colors=CLASS_COLORS, max_points=200000)
