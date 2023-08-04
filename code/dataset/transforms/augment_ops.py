"""
Random augmentation implemented by https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
"""
import math
import random
import re
import torch
import numpy as np
import PIL
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms.functional as F
import cv2
import math
# import imgaug.augmenters as iaa
from copy import deepcopy


class CustomAug(object):

    def __init__(self, p, version='le90'):
        self.p = p
        self.version = version

    def __call__(self, img):
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # print('called')
        # augments here
        proba = random.random()
        if proba <= self.p:
            img = self.adjust_contrast(img)
            # print('contrast aug')
        proba = random.random()
        if proba <= self.p:
            # pass
            # print(img.shape)
            img = self.apply_image(img, 0.2)
            # print('bright aug')
        proba = random.random()
        if proba <= self.p:
            k_level = 1
            img = self.random_blur(img, k_level=k_level)
            # print('blur')
        proba = random.random()
        if proba <= self.p:
            # print(img.shape)
            img = self.adjust_sharpeness(img)
            # print('sharp')
        proba = random.random()
        if proba <= self.p:
            self.hsv_augment(img)
            # print('hsv aug')
        proba = random.random()
        if proba <= self.p:
            img = self.GuassianNoise(img)

        # proba = random.random()
        # if proba <= self.p:
        #     angle = 10
        #     img = self.adjust_rotation(img, angle)

        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

    def apply_image(self, image, brightness):
        factor = 1.0 + random.uniform(-1.0 * brightness, brightness)
        table = np.array([(i / 255.0) * factor * 255 for i in np.arange(0, 256)]).clip(0, 255).astype(np.uint8)
        image = cv2.LUT(image, table)
        return image

    def hsv_augment(self, img, hgain=90, sgain=90, vgain=30):
        hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
        hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
        hsv_augs = hsv_augs.astype(np.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        # img_hsv.shape = (450, 720, 3)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    def random_blur(self, img, k_level):
        '''
        均值模糊
        :param img: 原始图片
        :param ksize: 模糊内核大小
        :return:
        '''
        k_level = 3 + random.randint(0, k_level)
        resultImg = cv2.blur(img, ksize=(k_level, k_level))
        return resultImg

    def adjust_contrast(self, img, level=0.4):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        factor = 1.0 + level * (1 - 2 * random.random())
        # img = mmcv.adjust_contrast(img, factor)
        enh_con = ImageEnhance.Contrast(img)
        img = enh_con.enhance(factor=factor)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

    def adjust_sharpeness(self, img, level=0.4):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        factor = 1.0 + level * (1 - 2 * random.random())
        enh_sha = ImageEnhance.Sharpness(img)
        img = enh_sha.enhance(factor=factor)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

    def adjust_rotation(self, img, angle):
        angle = random.uniform(-1.0, 1) * angle
        rows, cols = img.shape[:2]
        a, b = cols / 2, rows / 2
        M = cv2.getRotationMatrix2D((a, b), angle, 1)
        rotated_img = cv2.warpAffine(img, M, (cols, rows))  # 旋转后的图像保持大小不变
        return rotated_img

    def GuassianNoise(self, image, mu=0, sigma=0.08):
        """
        添加高斯噪声
        :param image: 输入的图像
        :param mu: 均值
        :param sigma: 标准差
        :return: 含有高斯噪声的图像
        """
        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mu, sigma * random.random(), image.shape)
        gauss_noise = image + noise
        if gauss_noise.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
        gauss_noise = np.uint8(gauss_noise * 255)
        return gauss_noise


_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])
_FILL = (128, 128, 128)
_DEFAULT_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


# define all kinds of functions


def _randomly_negate(v):
    return -v if random.random() > 0.5 else v


#################################################################################


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    kwargs_new = kwargs
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, expand=True, **kwargs_new)
    if _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(-rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix)
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs_new)
    return img.rotate(degrees, resample=kwargs['resample'])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=256, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(max(0, min(255, i + add)))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def distort(img, v, **__):
    w, h = img.size
    horizontal_tiles = int(0.1 * v)
    vertical_tiles = int(0.1 * v)

    width_of_square = int(math.floor(w / float(horizontal_tiles)))
    height_of_square = int(math.floor(h / float(vertical_tiles)))
    width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
    height_of_last_square = h - (height_of_square * (vertical_tiles - 1))
    dimensions = []

    for vertical_tile in range(vertical_tiles):
        for horizontal_tile in range(horizontal_tiles):
            if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([
                    horizontal_tile * width_of_square, vertical_tile * height_of_square,
                    width_of_last_square + (horizontal_tile * width_of_square),
                    height_of_last_square + (height_of_square * vertical_tile)
                ])
            elif vertical_tile == (vertical_tiles - 1):
                dimensions.append([
                    horizontal_tile * width_of_square, vertical_tile * height_of_square,
                    width_of_square + (horizontal_tile * width_of_square),
                    height_of_last_square + (height_of_square * vertical_tile)
                ])
            elif horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([
                    horizontal_tile * width_of_square, vertical_tile * height_of_square,
                    width_of_last_square + (horizontal_tile * width_of_square),
                    height_of_square + (height_of_square * vertical_tile)
                ])
            else:
                dimensions.append([
                    horizontal_tile * width_of_square, vertical_tile * height_of_square,
                    width_of_square + (horizontal_tile * width_of_square),
                    height_of_square + (height_of_square * vertical_tile)
                ])
    last_column = []
    for i in range(vertical_tiles):
        last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

    last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

    polygons = []
    for x1, y1, x2, y2 in dimensions:
        polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

    polygon_indices = []
    for i in range((vertical_tiles * horizontal_tiles) - 1):
        if i not in last_row and i not in last_column:
            polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

    for a, b, c, d in polygon_indices:
        dx = v
        dy = v

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
        polygons[a] = [x1, y1, x2, y2, x3 + dx, y3 + dy, x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
        polygons[b] = [x1, y1, x2 + dx, y2 + dy, x3, y3, x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
        polygons[c] = [x1, y1, x2, y2, x3, y3, x4 + dx, y4 + dy]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
        polygons[d] = [x1 + dx, y1 + dy, x2, y2, x3, y3, x4, y4]

    generated_mesh = []
    for idx, i in enumerate(dimensions):
        generated_mesh.append([dimensions[idx], polygons[idx]])
    return img.transform(img.size, PIL.Image.MESH, generated_mesh, resample=PIL.Image.BICUBIC)


def zoom(img, v, **__):
    #assert 0.1 <= v <= 2
    w, h = img.size
    image_zoomed = img.resize((int(round(img.size[0] * v)), int(round(img.size[1] * v))), resample=PIL.Image.BICUBIC)
    w_zoomed, h_zoomed = image_zoomed.size

    return image_zoomed.crop(
        (math.floor((float(w_zoomed) / 2) - (float(w) / 2)), math.floor((float(h_zoomed) / 2) - (float(h) / 2)),
         math.floor((float(w_zoomed) / 2) + (float(w) / 2)), math.floor((float(h_zoomed) / 2) + (float(h) / 2))))


def erase(img, v, **__):
    #assert 0.1<= v <= 1
    w, h = img.size
    w_occlusion = int(w * v)
    h_occlusion = int(h * v)
    if len(img.getbands()) == 1:
        rectangle = PIL.Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
    else:
        rectangle = PIL.Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion, len(img.getbands())) * 255))

    random_position_x = random.randint(0, w - w_occlusion)
    random_position_y = random.randint(0, h - h_occlusion)
    img.paste(rectangle, (random_position_x, random_position_y))
    return img


def skew(img, v, **__):
    #assert -1 <= v <= 1
    w, h = img.size
    x1 = 0
    x2 = h
    y1 = 0
    y2 = w
    original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]
    max_skew_amount = max(w, h)
    max_skew_amount = int(math.ceil(max_skew_amount * v))
    skew_amount = max_skew_amount
    new_plane = [
        (y1 - skew_amount, x1),  # Top Left
        (y2, x1 - skew_amount),  # Top Right
        (y2 + skew_amount, x2),  # Bottom Right
        (y1, x2 + skew_amount)
    ]
    matrix = []
    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(original_plane).reshape(8)
    perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
    perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)

    return img.transform(img.size,
                         PIL.Image.PERSPECTIVE,
                         perspective_skew_coefficients_matrix,
                         resample=PIL.Image.BICUBIC)


def blur(img, kernel_size, **__):
    #assert -3 <= v <= 3
    return img.filter(PIL.ImageFilter.GaussianBlur(kernel_size))


def hsv_augment(img, hgain, sgain, vgain, **__):
    hsv_augs = np.array([hgain, sgain, vgain])  # random gains
    # breakpoint()
    img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)  # no return needed
    return Image.fromarray(img, mode="RGB")


def gaussion_noise(image, mu, sigma, **__):
    image = np.array(image, dtype=float) / 255
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    gauss_noise = np.clip(gauss_noise, 1e-7, 1 - 1e-7)
    gauss_noise = np.uint8(gauss_noise * 256)
    return Image.fromarray(gauss_noise, mode="RGB")


def median_blur(image, ksize, **__):
    source = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    median = cv2.medianBlur(source, ksize)
    return Image.fromarray(median, mode="RGB")


#################################################################################


def rand_augment_transform(transforms, interpolation=None):
    return SequenceAugment([AutoAugmentOp(name, **transforms[name]) for name in transforms.keys()])


NAME_TO_OP = {
    'Distort': distort,
    'Zoom': zoom,
    'Blur': blur,
    'Skew': skew,
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'PosterizeOriginal': posterize,
    'PosterizeResearch': posterize,
    'PosterizeTpu': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
    'HSVAugment': hsv_augment,
    'GaussianNoise': gaussion_noise,
    'MedianBlur': median_blur
}


class AutoAugmentOp:

    def __init__(self, name, prob=0.5, **kwargs):
        self.aug_fn = NAME_TO_OP[name]
        self.prob = prob
        self.kwargs = kwargs
        self.kwargs['fillcolor'] = kwargs['img_mean'] if 'img_mean' in kwargs else _FILL
        self.kwargs['resample'] = kwargs['interpolation'] if 'interpolation' in kwargs else _DEFAULT_INTERPOLATION

    def __call__(self, img):
        kwargs = deepcopy(self.kwargs)
        for k in kwargs:
            if type(kwargs[k]) == dict and 'val' in kwargs[k]:
                if 'std' in kwargs[k]:
                    kwargs[k] = random.gauss(kwargs[k]['val'], kwargs[k]['std'])
                elif 'range' in kwargs[k]:
                    kwargs[k] = random.uniform(kwargs[k]['val'] - kwargs[k]['range'],
                                               kwargs[k]['val'] + kwargs[k]['range'])
        if random.random() > self.prob:
            return img
        return self.aug_fn(img, **kwargs)


class RandAugment:

    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        ops = np.random.choice(self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img


class SequenceAugment:

    def __init__(self, ops) -> None:
        self.ops = ops

    def __call__(self, img):
        for op in self.ops:
            img = op(img)
        return img


'''random erasing'''


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomResizedCropAndInterpolation:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio

        if interpolation == 'bilinear':
            self.interpolation = Image.BILINEAR
        elif interpolation == 'bicubic':
            self.interpolation = Image.BICUBIC
        elif interpolation == 'nearest':
            self.interpolation = Image.NEAREST

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, interpolation=self.interpolation)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(self,
                 probability=0.5,
                 min_area=0.02,
                 max_area=1 / 3,
                 min_aspect=0.3,
                 max_aspect=None,
                 mode='const',
                 min_count=1,
                 max_count=None,
                 num_splits=0,
                 device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(self.per_pixel,
                                                                     self.rand_color, (chan, h, w),
                                                                     dtype=dtype,
                                                                     device=self.device)
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (N, C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if img.ndim == 4:
            n = img.size(0)
            h = img.size(2)
            w = img.size(3)
        elif img.ndim == 3:
            n = 1
            h = img.size(1)
            w = img.size(2)

        mask = np.ones((n, h, w), np.float32)

        for i in range(n):
            for _ in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[i, y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).cuda()
        if img.ndim == 4:
            mask = mask.unsqueeze(1)
        #mask = mask.expand_as(img)
        img = img * mask

        return img


class Normalize:

    def __init__(self, mean, std, inplace=False, use_cuda=False):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        if use_cuda:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
        self.inplace = inplace

    def __call__(self, input):
        if self.inplace:
            # input[:, :3] = input[:, :3].sub_(self.mean).div_(self.std)
            input = input.unsqueeze(0)
            input[:, :3] = (input[:, :3] - self.mean) / self.std
            input = input.squeeze(0)
            # print(input.shape)
        else:
            input = (input - self.mean) / self.std
            input = input.squeeze(0)
        return input


class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class ConcatPositionEncoding:

    def __init__(self, uv, scale=224) -> None:
        self.scale = scale
        self.uv = uv

    def __call__(self, img: torch.Tensor):
        if self.uv:
            c, h, w = img.shape
            h_arr = torch.arange(h).reshape(h, 1).expand(h, w).unsqueeze(0) / self.scale
            w_arr = torch.arange(w).reshape(1, w).expand(h, w).unsqueeze(0) / self.scale
            img = torch.cat((img, h_arr, w_arr))
        return img
