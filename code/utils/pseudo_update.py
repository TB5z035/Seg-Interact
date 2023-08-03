import torch
import numpy as np
import os.path as osp
import logging
from tqdm import tqdm

from .misc import get_device, to_device, get_local_rank, save_pseudo_labels, save_pseudo_loss, seq_2_ordered_set

device = get_device()
logger = logging.getLogger('labeling_inference: inference')


def label_update(args, model, inf_loader, criterion, epoch):
    if get_local_rank() == 0:
        logger.info(f"Performing labeling inference {epoch}")
        model.eval()
        limit_dict = torch.load(
            osp.join(args.train_dataset['args']['root'], 'data_efficient', 'points',
                     f'points{args.train_dataset["args"]["limit"]}'))

        with torch.no_grad():
            for i, (inputs, labels, extras) in enumerate(tqdm(inf_loader)):
                _, inv_map = extras['maps']
                output = model(to_device(inputs, device))
                point_loss = criterion(output, to_device(labels, device))
                point_loss = point_loss[inv_map].cpu().numpy()
                gt_labels = labels[inv_map]

                preds = output.argmax(dim=1)
                preds = preds[inv_map]
                preds = preds.cpu().numpy()
                for pred in preds:
                    if pred >= 20:
                        pred = 255

                # Applying Inv_Mapping to Scene Names
                scenes_set = seq_2_ordered_set(extras['scene_id'])
                matching_index = np.arange(len(scenes_set))
                scenes = extras['scene_id']
                match_dict = dict(zip(scenes_set, matching_index))
                scenes = [match_dict.get(scene) for scene in scenes]
                scenes = torch.tensor(scenes)[inv_map]
                rev_dict = {v: k for k, v in match_dict.items()}
                scenes = [rev_dict.get(int(num)) for num in scenes]

                prev_scene_count = 0
                for scene in scenes_set:
                    this_scene_count = scenes.count(scene)
                    scene_loss = point_loss[prev_scene_count:prev_scene_count + this_scene_count]
                    scene_preds = preds[prev_scene_count:prev_scene_count + this_scene_count]
                    pred_label_ids = inf_loader.dataset.label_trainid_2_id(scene_preds)
                    scene_gt_labels = gt_labels[prev_scene_count:prev_scene_count + this_scene_count]
                    gt_label_ids = inf_loader.dataset.label_trainid_2_id(scene_gt_labels)
                    assert len(scene_loss) == len(pred_label_ids) == len(
                        gt_label_ids), 'labeling inference dimensions mismatch'
                    save_pseudo_labels(np.stack((pred_label_ids, gt_label_ids)), args.train_dataset['args']['root'],
                                       scene, epoch)
                    save_pseudo_loss(scene_loss, args.train_dataset['args']['root'], scene, epoch)
                    if epoch == 0:
                        limit_mask = np.ones_like(np.arange(this_scene_count), dtype=bool)
                        limit = limit_dict[scene]
                        limit_mask[limit] = False
                        prev_labels = gt_labels[prev_scene_count:prev_scene_count + this_scene_count]
                        prev_labels[limit_mask] = 255
                        prev_label_ids = inf_loader.dataset.label_trainid_2_id(prev_labels)
                        assert len(prev_label_ids) == len(
                            gt_label_ids), 'labeling inference dimensions mismatch at epoch 0'
                        np.save(
                            osp.join(args.train_dataset['args']['root'], 'scans', scene,
                                     f'{scene}_updated_labels_iter_{epoch-1}.npy'), prev_label_ids)
                    prev_scene_count += this_scene_count


def get_n_update_count(count_file_path: str, reset: bool):
    assert osp.exists(count_file_path), f'path to save count file {count_file_path} does not exist'
    count_path = osp.join(count_file_path, 'inference_count.npy')
    if osp.exists(count_path) and not reset:
        current_count = np.load(count_path) + 1
        np.save(count_path, current_count)
        return int(current_count)
    else:
        np.save(count_path, np.array(0))
        return 0
