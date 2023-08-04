import logging

import numpy as np
import torch
from tqdm import tqdm

from .misc import (get_device, get_local_rank, save_pseudo_labels,
                   save_pseudo_loss, to_device)

device = get_device()
logger = logging.getLogger('pseudo_label_update')


def label_update(args, model, train_loader, criterion, epoch):
    if get_local_rank() == 0:
        logger.info(f"Performing pseudo labeling inference {str(epoch)}")
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels, extras) in enumerate(tqdm(train_loader)):
                output = model(to_device(inputs, device))
                gt_labels = extras['gt_labels']
                point_loss = criterion(output, to_device(gt_labels, device)).cpu()
                point_loss = point_loss.numpy()

                _, preds = torch.topk(output, 1)
                preds = torch.squeeze(preds.t()).cpu()
                preds = preds.numpy()
                for pred in preds:
                    if pred >= 20:
                        pred = 255

                scenes = set(extras['scene_ids'])
                prev_scene_count = 0
                for scene in scenes:
                    this_scene_count = extras['scene_ids'].count(scene)
                    scene_preds = preds[prev_scene_count:prev_scene_count + this_scene_count]
                    scene_loss = point_loss[prev_scene_count:prev_scene_count + this_scene_count]
                    label_ids = train_loader.dataset.label_trainid_2_id(scene_preds)
                    save_pseudo_labels(label_ids, args.train_dataset['args']['root'], scene, epoch)
                    save_pseudo_loss(scene_loss, args.train_dataset['args']['root'], scene, epoch)
                    prev_scene_count += this_scene_count


# if __name__ == "__main__":
