import random
import logging
import os.path as osp

# import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import tensorboardX

from ..dataset import DATASETS
# from .metrics import IoU, mIoU, Acc, MATRICS
from ..metrics import METRICS
from ..network import NETWORKS
from ..vis_and_lab.point_selection import highest_loss_filtering
from ..vis_and_lab import prep_files_for_visuaization

from ..optimizer import OPTIMIZERS, SCHEDULERS
from .args import get_args
from .misc import get_device, init_directory, init_logger, to_device, get_local_rank, get_world_size, save_checkpoint, clean_inference_paths, clean_prev_inf_paths, clean_vis_paths
from .validate import validate
from .pseudo_update import label_update, get_n_update_count

device = get_device()
logger = logging.getLogger('train')


def train(local_rank=0, world_size=1, args=None):
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
    train_dataset = DATASETS[args.train_dataset['name']](**(args.train_dataset['args'] | {
        'labeling_inference': args.labeling_inference,
        'inference_save_path': args.inference_save_path
    }))
    val_dataset = DATASETS[args.val_dataset['name']](**args.val_dataset['args'])
    inf_dataset = DATASETS[args.inf_dataset['name']](**args.inf_dataset['args'])
    assert train_dataset.num_channel == val_dataset.num_channel
    assert train_dataset.num_train_classes == val_dataset.num_train_classes
    logger.info(
        f"Train dataset: {args.train_dataset}\nVal dataset: {args.val_dataset}\nInf dataset: {args.inf_dataset}")
        
    # DataLoader
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=world_size,
                                                                        rank=local_rank,
                                                                        shuffle=True)
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.train_num_workers,
        sampler=train_sampler,
        collate_fn=train_dataset._collate_fn if hasattr(train_dataset, '_collate_fn') else None)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                shuffle=False,
                                num_workers=args.val_num_workers,
                                collate_fn=val_dataset._collate_fn)
    inf_dataloader = DataLoader(inf_dataset,
                                batch_size=args.val_batch_size,
                                shuffle=False,
                                num_workers=args.val_num_workers,
                                collate_fn=inf_dataset._collate_fn)
    logger.info(
        f"Train dataloader: {len(train_dataloader)}\nVal dataloader: {len(val_dataloader)}\nInf dataloader: {len(inf_dataloader)}"
    )

    # Model
    network = NETWORKS[args.model['name']](train_dataset.num_channel, train_dataset.num_train_classes)
    network = network.to(device)
    # Load pretrained model
    if args.resume:
        logger.info(f"Resume training from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        network.load_state_dict(ckpt['network'])
    elif args.model['args']['pretrained']:
        logger.info(f"Load pretrained model from {args.pretrained}")
        ckpt = torch.load(args.model['args']['pretrained'], map_location=device)
        network.load_state_dict(ckpt['network'])
        pass
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
    logger.info(f"Model: {args.model}")

    # Optimizer
    optimizer = OPTIMIZERS[args.optimizer['name']](network.parameters(), **args.optimizer['args'])
    if args.resume:
        optimizer.load_state_dict(ckpt['optimizer'])

    # Criterion
    # TODO: init from args
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    point_criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    # Scheduler
    scheduler = SCHEDULERS[args.scheduler['name']](optimizer, **args.scheduler['args'])
    if args.resume:
        scheduler.load_state_dict(ckpt['scheduler'])

    if args.resume:
        global_iter = [ckpt['iter']]
        start_epoch = ckpt['epoch'] + 1
    else:
        global_iter = [0]
        start_epoch = 0

    # Labeling Inference Init
    if args.labeling_inference:
        inference_count = get_n_update_count(args.inference_count_path, reset=args.inference_count_reset)
        clean_inference_paths(args.inference_save_path)
        # Visualization Init
        if args.visualize:
            clean_vis_paths(args.vis_save_path)

    for epoch_idx in range(start_epoch, args.epochs):
        # Train
        train_one_epoch(network,
                        optimizer,
                        train_dataloader,
                        criterion,
                        epoch_idx,
                        global_iter,
                        scheduler=scheduler,
                        val_loader=val_dataloader,
                        writer=writer)

        # Validate
        if epoch_idx % args.val_epoch_freq == 0:
            validate(network,
                     val_dataloader,
                     criterion,
                     metrics=[METRICS[metric] for metric in args.metrics],
                     global_iter=global_iter[0],
                     writer=writer)
            #torch.cuda.empty_cache()

        #if epoch_idx % args.save_epoch_freq == 0:
        #save_checkpoint(network, args, epoch_idx, global_iter[0], optimizer, scheduler, name=f'epoch#{epoch_idx}')

        # Labeling Inference
        if args.labeling_inference and epoch_idx % args.labeling_inference_epoch == 0:
            label_update(args, network, inf_dataloader, point_criterion, inference_count)
            highest_loss_filtering(args,
                                   args.inference_save_path,
                                   inference_count,
                                   visualize=args.visualize,
                                   vis_path=args.vis_save_path)
            inference_count = get_n_update_count(args.inference_count_path, reset=False)
            clean_prev_inf_paths(args.inference_save_path)
        # if epoch_idx != 0 and epoch_idx % 5 == 0:
        #     print(torch.cuda.memory_summary())
        #     torch.cuda.empty_cache()
        #     print(torch.cuda.memory_summary())

    # Visualize
    if args.visualize:
        label_update(args, network, inf_dataloader, point_criterion, 'final')
        prep_files_for_visuaization(inf_dataset, args.inference_save_path, args.vis_save_path, args.visualize)
    #save_checkpoint(network, args, epoch_idx=None, iter_idx=None, optimizer=None, scheduler=None, name=f'last')


def train_one_epoch(model,
                    optimizer,
                    train_loader,
                    criterion,
                    epoch_idx,
                    iter_idx,
                    logging_freq=10,
                    scheduler=None,
                    val_loader=None,
                    writer=None):
    model.train()

    for i, (inputs, labels, extras) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(to_device(inputs, device))
        loss = criterion(output, to_device(labels, device))
        loss.backward()
        optimizer.step()
        if get_world_size() > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= get_world_size()
        if i % logging_freq == 0:
            logger.info(
                f"Epoch: {epoch_idx:4d}, Iteration: {i:4d} / {len(train_loader):4d} [{iter_idx[0]:5d}], Loss: {loss.item()}"
            )
        if writer is not None:
            writer.add_scalar('train/loss', loss.item(), iter_idx[0])
        iter_idx[0] += 1
    if scheduler is not None:
        scheduler.step()

    #del loss
    #del output


if __name__ == '__main__':
    args, args_text = get_args()
    init_directory(args, args_text)
    args.port = random.randint(10000, 20000)
    logger.info(args_text)

    if args.world_size == 1:
        train(args=args)
    else:
        mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))
