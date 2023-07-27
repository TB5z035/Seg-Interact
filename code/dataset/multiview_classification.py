
import os
import os.path as osp
import re
from collections import namedtuple
import logging
from PIL import Image
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
from .transforms import multiview_transform
from . import register_dataset

is_valid_scene_id = re.compile(r'^scene\d{4}_\d{2}$').match
logger = logging.getLogger('multiview')

@register_dataset('multiview')
class MultiviewDataset(Dataset):
    """
    Multiview dataset
    The structure of ScanNet Dataset:

    DATASET_ROOT
    ├── train 
    │   ├── label_01
    │   │   ├── item_01
    │   │   │   ├── view_0.png
    │   │   │   ├── view_1.png
    │   │   │   ├── ...
    │   │   │   └── view_4.png
    │   │   ├── ...      
    │   │   ...
    │   ├── label_02
    │   ├── ...
    │   └── label_60(maybe)
    └── test
        └── (the same)

    """
    @property
    def num_train_classes(self):
        return len(self.classes)
    
    @property
    def num_channel(self):
        return 3
    
    @property
    def train_class_names(self):
        return self.classes
    
    def ignore_class(self):
        return None
    
    def find_classes(self, classes):
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return class_to_idx

    def __init__(self,
                 root,
                 classes,
                 split=None,
                 view_num=5,
                 augmentation=None,
                 UV=None,
                 remode=None,
                 reprob=None,
                 interpolation=None,
                 transform=None,
                 target_transform=None,
                 Cutmix=None,
                 random_drop=None,
                 random_cut=None,
                 random_match_wrong=None,
                 view_shuffle=None,
                 view_rotate=None,
                 size_rand=None,
                 **kargs):
        self.x = []
        self.y = []
        if split == 'train':
            transform = multiview_transform.build_train_transforms(augmentation, UV, reprob, remode, interpolation, inplace=True)[0]
        else:
            transform = multiview_transform.build_val_transforms(UV, interpolation)[0]
        self.root = os.path.join(root, split)
        root = self.root
        self.classes = classes
        self.class_to_idx = self.find_classes(classes)

        self.transform = transform
        self.target_transform = target_transform

        # --- ll start
        if view_shuffle is not None:
            self.view_shuffle_prob = view_shuffle['prob']
        else:
            self.view_shuffle_prob = 0
        if view_rotate is not None:
            self.view_rotate_prob = view_rotate['prob']
        else:
            self.view_rotate_prob = 0
        if size_rand is not None:
            self.size_rand_prob = size_rand['prob']
        else:
            self.size_rand_prob = 0

        if Cutmix is not None:
            self.cutmix_prob = Cutmix['prob']
            self.cutmix_range = Cutmix['range']
        else:
            self.cutmix_prob = 0

        if random_drop is not None:
            self.random_drop_prob = random_drop['prob']
        else:
            self.random_drop_prob = 0

        if random_cut is not None:
            self.random_cut_prob = random_cut['prob']
        else:
            self.random_cut_prob = 0

        if random_match_wrong is not None:
            self.random_match_wrong_prob = random_match_wrong['prob']
        else:
            self.random_match_wrong_prob = 0

        # --- ll end

        # root<train/test> / <label>  / <item> / <view>.png
        for label in self.classes:  # Label
            if not os.path.exists(os.path.join(root, label)):
                continue
            for item in os.listdir(os.path.join(root, label)):
                views = []
                for view in range(view_num):
                    views.append(os.path.join(root, label, item, str(view) + '.png'))

                views = sorted(views)
                # views.reverse()
                # print(views)

                self.x.append(views)
                self.y.append(self.class_to_idx[label])

    def __getitem__(self, index):
        orginal_views = self.x[index]
        # print('orginal_views', len(orginal_views))
        views_list = []
        sizes = []

        # shuffle views
        shuffle_score = np.random.rand(1)
        if shuffle_score <= self.view_shuffle_prob:
            reorder_views = copy.deepcopy(orginal_views[1:])
            random.shuffle(reorder_views)
            orginal_views = [orginal_views[0]]
            orginal_views.extend(reorder_views)

        # this item already finish cutmix or drop views?
        flag = False
        # which view
        idx = -1
        drop_prob = np.random.rand(1)
        drop_samples = random.sample(range(1, 5), random.randint(1, 4))

        for view in orginal_views:
            idx += 1
            try:
                im = Image.open(view)
                if idx == 0:
                    size = im.size
                else:
                    size = (224, 224)
                im = im.resize((224, 224))
            except:
                im = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                size = (224, 224)

            sizes.append(size)
            im = im.resize((224, 224))
            im = im.convert('RGB')

            # match_wrong
            prob = np.random.rand(1)
            if prob < self.random_match_wrong_prob and idx != 0:
                im, sizes = self.Match_wrong(im, idx, sizes)

            # rotate
            prob = np.random.rand(1)
            if prob < self.view_rotate_prob and idx != 0:
                r_method = random.choice(range(3))
                if r_method == 0:
                    im = im.transpose(Image.ROTATE_90)
                    sizes[idx] = (sizes[idx][1], sizes[idx][0])
                elif r_method == 1:
                    im = im.transpose(Image.ROTATE_180)
                elif r_method == 2:
                    im = im.transpose(Image.ROTATE_270)
                    sizes[idx] = (sizes[idx][1], sizes[idx][0])
            # im.save('tmp.png')

            # size rand reverse
            prob = np.random.rand(1)
            if prob < self.size_rand_prob and idx != 0:
                sizes[idx] = (sizes[idx][1], sizes[idx][0])
            # cut
            prob = np.random.rand(1)
            if prob < self.random_cut_prob and idx != 0:
                im, sizes = self.Cut(im, idx, sizes)

            # cut_mix
            prob = np.random.rand(1)
            if not flag and idx != 0:
                if prob < self.cutmix_prob:
                    im = self.Cut_mix(im, idx, self.cutmix_range)
                    # flag = True

            # drop
            if drop_prob < self.random_drop_prob and idx != 0:
                if idx in drop_samples:
                    im = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                    sizes[idx] = (0, 0)

            if self.transform is not None:
                im = self.transform(im)
            views_list.append(im)
        views = np.stack(views_list)
        views = torch.from_numpy(views)

        # sizes = np.array(sizes)
        # sizes = torch.from_numpy(sizes).float()

        return views, self.y[index], []

    def Cut_mix(self, img, view_idx, range):
        # original image
        img_A = np.array(img)
        label_idx = np.random.randint(0, len(os.listdir(self.root)))
        itm_num = len(os.listdir(os.path.join(self.root, os.listdir(self.root)[label_idx])))

        # random image (which has the same view with the original image)
        img_B_views = self.x[np.random.randint(0, itm_num)]

        try:
            img_B = Image.open(img_B_views[view_idx])
            img_B = np.array(img_B.resize((224, 224)).convert('RGB'))
        except:
            img_B = img_A

        cut_rat = np.random.uniform(0, range)

        bbx1, bby1, bbx2, bby2 = self.rand_bbox((224, 224), cut_rat)

        img_A[bbx1:bbx2, bby1:bby2, :] = img_B[bbx1:bbx2, bby1:bby2, :]

        img_A = Image.fromarray(img_A)

        return img_A

    def rand_bbox(self, size, cut_rat):
        W = size[0]
        H = size[1]
        cut_rat = np.sqrt(cut_rat)
        cut_w = np.int_(W * cut_rat)
        cut_h = np.int_(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        #限制坐标区域不超过样本大小

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def Match_wrong(self, img, idx, sizes):
        img_A = img
        label_idx = np.random.randint(0, len(os.listdir(self.root)))
        itm_num = len(os.listdir(os.path.join(self.root, os.listdir(self.root)[label_idx])))

        # random image (which has the same view with the original image)
        img_B_views = self.x[np.random.randint(0, itm_num)]

        try:
            img_B = Image.open(img_B_views[idx])
            new_size = img_B.size
        except:
            img_B = img_A
            new_size = sizes[idx]

        img_A = img_B.resize((224, 224)).convert('RGB')

        sizes[idx] = new_size

        return img_A, sizes

    def Cut(self, img, idx, sizes):
        img = np.array(img)
        size = sizes[idx]

        cut_rat = np.random.uniform(0.4, 1)
        start_, _, end_, _ = self.rand_bbox((224, 224), cut_rat)

        img = Image.fromarray(img[start_:end_, start_:end_, :])

        img = img.resize((224, 224)).convert('RGB')

        rat = (end_ - start_) / 224

        w = int(size[0] * rat)
        h = int(size[1] * rat)

        sizes[idx] = (w, h)

        return img, sizes

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
