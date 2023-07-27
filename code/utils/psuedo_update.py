import MinkowskiEngine as ME
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging

from .misc import get_device, to_device, get_local_rank, save_pseudo_labels, save_pseudo_loss

device = get_device()
logger = logging.getLogger('pseudo_label_update')


def label_update(args, model, train_loader, criterion, epoch_idx):
    if get_local_rank() == 0:
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels, extras) in enumerate(train_loader):
                output = model(to_device(inputs, device))
                point_loss = criterion(output, to_device(labels, device)).cpu()
                point_loss = point_loss.numpy()

                _, preds = torch.topk(output, 1)
                preds = torch.squeeze(preds.t()).cpu()
                preds = preds.numpy()
                for pred in preds:
                    if pred == 20:
                        pred = 255

                scenes = set(extras)
                prev_scene_count = 0
                for scene in scenes:
                    this_scene_count = extras.count(scene)
                    scene_preds = preds[prev_scene_count:prev_scene_count+this_scene_count]
                    scene_loss = point_loss[prev_scene_count:prev_scene_count+this_scene_count]
                    label_ids = train_loader.dataset.label_trainid_2_id(scene_preds)
                    save_pseudo_labels(label_ids, args.train_dataset['args']['root'], scene, epoch_idx)
                    save_pseudo_loss(scene_loss, args.train_dataset['args']['root'], scene, epoch_idx)
                    prev_scene_count += this_scene_count


# if __name__ == "__main__":

