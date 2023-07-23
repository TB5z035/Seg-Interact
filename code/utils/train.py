import random
import logging

import MinkowskiEngine as ME
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
import tensorboardX

from ..dataset import DATASETS
from .metrics import IoU, mIoU
from ..network import NETWORKS
from .args import get_args
from .misc import get_device, init_directory, init_logger, to_device, get_local_rank, get_world_size, get_time_str, save_checkpoint
from .validate import validate

device = get_device()
logger = logging.getLogger('train')


def train(local_rank=0, world_size=1, args=None):
    # Sanity check
    assert args is not None
    logger.warning(f"Local rank: {local_rank}, World size: {world_size}")

    # TODO: reproducibility

    # Distributed init
    if world_size > 1:
        dist.init_process_group(backend='nccl',
                                rank=local_rank,
                                world_size=world_size,
                                init_method=f'tcp://localhost:{args.port}')
        torch.cuda.set_device(local_rank)
    logger.info(f"torch.distributed initialized: {dist.is_initialized()}")
    init_logger(args)
    writer = args.writer if get_local_rank() == 0 else None

    # Dataset
    train_dataset = DATASETS[args.train_dataset](args.train_dataset_root, split='train', transform=args.train_transform)
    val_dataset = DATASETS[args.val_dataset](args.val_dataset_root, split='val', transform=args.val_transform)
    assert train_dataset.num_channel == val_dataset.num_channel
    assert train_dataset.num_train_classes == val_dataset.num_train_classes
    logger.info(f"Train dataset: {args.train_dataset}, Val dataset: {args.val_dataset}")

    # DataLoader
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=world_size,
                                                                        rank=local_rank,
                                                                        shuffle=True)
    else:
        train_sampler = None
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=(train_sampler is None),
                                  num_workers=args.train_num_workers,
                                  sampler=train_sampler,
                                  collate_fn=train_dataset._collate_fn)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                shuffle=False,
                                num_workers=args.val_num_workers,
                                collate_fn=val_dataset._collate_fn)
    logger.info(f"Train dataloader: {len(train_dataloader)}, Val dataloader: {len(val_dataloader)}")

    # Model
    network = NETWORKS[args.model](train_dataset.num_channel, train_dataset.num_train_classes)
    network = network.to(device)
    # Load pretrained model
    if args.resume:
        logger.info(f"Resume training from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        network.load_state_dict(ckpt['network'])
    elif args.pretrained:
        logger.info(f"Load pretrained model from {args.pretrained}")
        ckpt = torch.load(args.resume, map_location=device)
        network.load_state_dict(ckpt['network'])
        pass
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
    logger.info(f"Model: {args.model}")

    # Optimizer
    # TODO: init from args
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    if args.resume:
        optimizer.load_state_dict(ckpt['optimizer'])

    # Criterion
    # TODO: init from args
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    if args.resume:
        global_iter = [ckpt['iter']]
        start_epoch = ckpt['epoch'] + 1
    else:
        global_iter = [0]
        start_epoch = 0

    for epoch_idx in range(start_epoch, args.epochs):
        train_one_epoch(network,
                        optimizer,
                        train_dataloader,
                        criterion,
                        epoch_idx,
                        global_iter,
                        val_loader=val_dataloader,
                        writer=writer)
        # Validate
        if epoch_idx % args.val_epoch_freq == 0:
            validate(network, val_dataloader, criterion, metrics=[mIoU, IoU], global_iter=global_iter[0], writer=writer)
        if epoch_idx % args.save_epoch_freq == 0:
            save_checkpoint(network, args, epoch_idx, global_iter[0], optimizer, scheduler=None, name=f'epoch#{epoch_idx}')

    save_checkpoint(network, args, epoch_idx=None, global_iter=None, optimizer=None, scheduler=None, name=f'last')


def train_one_epoch(model,
                    optimizer,
                    train_loader,
                    criterion,
                    epoch_idx,
                    iter_idx,
                    logging_freq=10,
                    val_loader=None,
                    writer=None):
    model.train()
    for i, (inputs, labels, _) in enumerate(train_loader):
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


if __name__ == '__main__':
    args, args_text = get_args()
    init_directory(args, args_text)
    args.port = random.randint(10000, 20000)
    logger.info(args_text)

    if args.world_size == 1:
        train(args=args)
    else:
        mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))