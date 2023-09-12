# Necessary for advanced config parsing with hydra and omegaconf
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)

import torch
from .sp_visualization import show
from .dataset.s3dis import CLASS_NAMES, CLASS_COLORS
from .dataset.s3dis import S3DIS_NUM_CLASSES as NUM_CLASSES
from .dataset.sp_transforms import *
from .dataset.s3dis import sp_S3DIS
from .dataset.superpoint_base import sp_init

from .utils.args import get_args


cfg, _ = get_args()
cfg_dm = cfg.datamodule

dataset = sp_init(cfg_dm, sp_S3DIS)
print(dataset)


# 6 areas of s3dis dataset, nag_[i] is a NAG object of that area
nag_0 = dataset[0]
nag_1 = dataset[1]
nag_2 = dataset[2]
nag_3 = dataset[3]
nag_4 = dataset[4]
nag_5 = dataset[5]
nag = nag_0
print(nag)

xyz = nag_0[0].pos
labels = nag_0[0].y.argmax(1)

super_indices = nag.get_super_index(3,0)
indices = torch.where(super_indices == 0)
print(labels[indices])

#Full Vis
'''
show(
    nag, 
    class_names=CLASS_NAMES, 
    ignore=NUM_CLASSES,
    class_colors=CLASS_COLORS,
    max_points=200000
)
'''

#Partial Vis
# Pick a center and radius for the spherical sample
center = center = torch.tensor([[-2, 18, 3]]).to(nag.device)
radius = 5

# Create a mask on level-0 (ie points) to be used for indexing the NAG 
# structure
mask = torch.where(torch.linalg.norm(nag[0].pos - center, dim=1) < radius)[0]

# Subselect the hierarchical partition based on the level-0 mask
nag_visu = nag.select(0, mask)

# Visualize the sample
show(
    nag_visu,
    class_names=CLASS_NAMES,
    ignore=NUM_CLASSES,
    class_colors=CLASS_COLORS, 
    max_points=100000
)
