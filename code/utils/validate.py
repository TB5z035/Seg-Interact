import logging
import os.path as osp
import random

import MinkowskiEngine as ME
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from ..dataset import DATASETS
from ..network import NETWORKS
from ..optimizer import OPTIMIZERS, SCHEDULERS
from ..policy.mixture_filter import mixture_filter, mixture_filter_trivial
from .args import get_args
from .metrics import IoU, mIoU
from .misc import (clear_paths, get_device, get_local_rank, get_time_str,
                   get_world_size, init_directory, init_logger,
                   save_checkpoint, to_device)
from .psuedo_update import label_update

device = get_device()
logger = logging.getLogger('validate')

from tqdm import tqdm


def validate(model, val_loader: DataLoader, criterion=None, metrics=[], writer=None, global_iter=None):
    if get_local_rank() == 0:
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            metric_objs = [
                metric(
                    val_loader.dataset.num_train_classes,
                    val_loader.dataset.ignore_class,
                    val_loader.dataset.train_class_names,
                ) for metric in metrics
            ]

            for idx, (inputs, labels, extras) in enumerate(tqdm(val_loader)):
                output = model(to_device(inputs, device))
                output = output[extras['maps'][1]]
                pred = output.argmax(dim=1).cpu()
                if 'maps' in extras:
                    labels = labels[extras['maps'][1]]

                if criterion is not None:
                    loss_sum += criterion(output, to_device(labels, device)).item()

                for metric in metric_objs:
                    metric.record(pred, labels)

        loss_sum /= len(val_loader)

        metric_results = {}
        for metric in metric_objs:
            metric_results[metric.NAME] = metric.calc()
            metric.log(logger, writer=writer, global_iter=global_iter, name_prefix='val/')

        return loss_sum, metric_results
    else:
        return 0, {}


def unc_inference(model, val_loader: DataLoader, criterion=None, metrics=[], writer=None, global_iter=None):
    """
    This function should be merged with validate() in the future.
    The loss function and metrics and save should be implemented as callbacks.
    """

    if get_local_rank() == 0:
        model.eval()

        results = {}
        with torch.no_grad():
            for idx, (inputs, _, extras) in enumerate(tqdm(val_loader, desc='Inference')):
                # FIXME ugly implementation
                pred, uncertainty = to_device(model.module.unc_infer(to_device(inputs, device)), 'cpu')
                if 'maps' in extras:
                    pred = pred[extras['maps'][1]]
                    uncertainty = uncertainty[extras['maps'][1]]

                for batch_idx, scene_path in enumerate(extras['scene_path']):
                    mask = (extras['batch_indices'] == batch_idx).squeeze()
                    if 'maps' in extras:
                        mask = mask[extras['maps'][1]]
                    results[extras['scene_path'][batch_idx]] = (pred[mask], uncertainty[mask])

            b_pred = torch.cat([item[0] for item in results.values()], dim=0)
            b_uncertainty = torch.cat([item[1] for item in results.values()], dim=0)
            np.save(
                '/hdd1/Guest/tb5zhh/workspace/pred.npy',
                b_pred.numpy(),
            )
            np.save(
                '/hdd1/Guest/tb5zhh/workspace/uncertainty.npy',
                b_uncertainty.numpy(),
            )
            filter_fns = [
                mixture_filter_trivial(np.asarray(b_uncertainty[b_pred == i]),
                                       'gamma',
                                       num_trial=5,
                                       save_dir='/hdd1/Guest/tb5zhh/workspace/Seg-Interact/save/',
                                       caption=f'{i}')[1] for i in range(val_loader.dataset.num_train_classes)
            ]
            for scene_path, (pred, unc) in tqdm(results.items(), desc='Save'):
                for cate_idx, filter_fn in enumerate(filter_fns):
                    pred[pred == cate_idx][filter_fn(unc[pred == cate_idx])] = 255
                val_loader.dataset.save_pred(scene_path,
                                             val_loader.dataset.label_trainid_2_id(pred),
                                             pred,
                                             save_root=osp.join(args.exp_dir, 'pseudo', args.start_time))

        return 0, {}
    else:
        return 0, {}


def run_validate(local_rank=0, world_size=1, args=None):
    # Sanity check
    assert args is not None
    logger.warning(f"Local rank: {local_rank}, World size: {world_size}")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Distributed init
    # if world_size > 1:
    dist.init_process_group(backend='nccl',
                            rank=local_rank,
                            world_size=world_size,
                            init_method=f'tcp://localhost:{args.port}')
    torch.cuda.set_device(local_rank)
    logger.info(f"torch.distributed initialized: {dist.is_initialized()}")
    writer = init_logger(args)

    # Dataset
    val_dataset = DATASETS[args.val_dataset['name']](**args.val_dataset['args'])
    logger.info(f"Val dataset: {args.val_dataset}")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                shuffle=False,
                                num_workers=args.val_num_workers,
                                collate_fn=val_dataset._collate_fn)
    logger.info(f"Val dataloader: {len(val_dataloader)}")

    # Model
    network = NETWORKS[args.model](val_dataset.num_channel, val_dataset.num_train_classes)
    network = network.to(device)
    # Load pretrained model
    if args.pretrained:
        logger.info(f"Load pretrained model from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        network.load_state_dict(ckpt['network'])
        pass
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
    logger.info(f"Model: {args.model}")

    action = {
        'validate': validate,
        'unc_inference': unc_inference,
    }[args.action]

    action(network, val_dataloader, writer=writer, global_iter=0, criterion=None, metrics=[IoU, mIoU])


if __name__ == '__main__':
    args, args_text = get_args()
    init_directory(args, args_text)
    args.port = random.randint(10000, 20000)
    logger.info(args_text)

    assert args.pretrained is not None, "Pretrained model is required for validation"

    if args.world_size == 1:
        run_validate(args=args)
    else:
        mp.spawn(run_validate, nprocs=args.world_size, args=(args.world_size, args))
