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

nag = dataset[0]
num_points = nag.num_points
print(num_points)
labels = nag[0].y.argmax(1)
level = 3
sup_inds = nag.get_super_index(level, 0)
sup_max = num_points[level]
'''
mode_num = []
percentage = []

for i in range(sup_max):
    inds = torch.where(sup_inds == i)[0]
    sup_labels = labels[inds]
    main_num = torch.mode(sup_labels)[0]
    main_freq = torch.bincount(sup_labels)[main_num]
    percent = main_freq / len(sup_labels)

    mode_num.append(main_num)
    percentage.append(percent)

mode_num = torch.tensor(mode_num)
percentage = torch.tensor(percentage)

print(mode_num)
print(percentage)
print(f'average contain mode_num percentage: {torch.sum(percentage)/len(percentage)}')
print(f'average pure mode_num percentage: {len(percentage[percentage == 1.])/len(percentage)}')
'''

# show(nag, class_names=CLASS_NAMES, ignore=NUM_CLASSES, class_colors=CLASS_COLORS, max_points=200000)
