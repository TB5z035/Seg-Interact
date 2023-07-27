from . import register_transform, Transform
import torchvision.transforms as transforms
from PIL import Image
from . import augment_ops

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

BRICK_DATASET_MEAN = (0.6275333, 0.5790175, 0.58084536)
BRICK_DATASET_STD = (0.26300925, 0.23415063, 0.2092097)

CIFAR_DEFAULT_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR_DEFAULT_STD = (0.24703233, 0.24348505, 0.26158768)


def build_train_transforms(transforms_config, uv, reprob=0., remode='pixel', interpolation=None, inplace=False):
    trans_l = [
        augment_ops.rand_augment_transform(transforms_config, interpolation),
        transforms.ToTensor(),
        augment_ops.ConcatPositionEncoding(uv['concat'], scale=224),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        augment_ops.Normalize(mean=BRICK_DATASET_MEAN, std=BRICK_DATASET_STD, inplace=inplace),
    ]
    trans_r = [
        # augment_ops.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
    if reprob > 0:
        trans_r.append(augment_ops.RandomErasing(reprob, mode=remode, max_count=1, num_splits=0, device='cuda'))
    return transforms.Compose(trans_l), transforms.Compose(trans_r)


def build_val_transforms(uv, interpolation='bilinear', inplace=False):
    trans_l = [
        transforms.ToTensor(),
        augment_ops.ConcatPositionEncoding(uv['concat'], scale=224),
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        augment_ops.Normalize(mean=BRICK_DATASET_MEAN, std=BRICK_DATASET_STD, inplace=inplace),
    ]
    trans_r = [
        # augment_ops.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
    return transforms.Compose(trans_l), transforms.Compose(trans_r)
