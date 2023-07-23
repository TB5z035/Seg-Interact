import random
import logging

import MinkowskiEngine as ME
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist

from ..dataset import DATASETS
from .metrics import IoU, mIoU
from ..network import NETWORKS
from .args import get_args
from .misc import get_device, init_directory, init_logger, to_device
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
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size, init_method=f'tcp://localhost:{args.port}')
        torch.cuda.set_device(local_rank)
    logger.info(f"torch.distributed initialized: {dist.is_initialized()}")
    init_logger(args)

    # Dataset
    train_dataset = DATASETS[args.train_dataset](args.train_dataset_root, split='train', transform=args.train_transform)
    val_dataset = DATASETS[args.val_dataset](args.val_dataset_root, split='val', transform=args.val_transform)
    assert train_dataset.num_channel == val_dataset.num_channel
    assert train_dataset.num_train_classes == val_dataset.num_train_classes
    logger.info(f"Train dataset: {args.train_dataset}, Val dataset: {args.val_dataset}")

    # DataLoader
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
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
    # Load pretrained model
    # TODO: Implement this
    network = network.to(device)
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
    logger.info(f"Model: {args.model}")

    # Optimizer
    # TODO: init from args
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # Criterion
    # TODO: init from args
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    global_iter = 0
    for epoch_idx in range(args.epochs):
        train_one_epoch(network, optimizer, train_dataloader, criterion, epoch_idx, global_iter, val_loader=val_dataloader)
        # Validate
        val_loss, val_metrics = validate(network, val_dataloader, criterion, metrics=[mIoU, IoU])

    # Save model
    ...


def train_one_epoch(model, optimizer, train_loader, criterion, epoch_idx, iter_idx, logging_freq=10, val_loader=None):
    model.train()
    for i, (inputs, labels, _) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(to_device(inputs, device))
        loss = criterion(output, to_device(labels, device))
        loss.backward()
        optimizer.step()
        if i % logging_freq == 0:
            logger.info(f"Epoch: {epoch_idx:4d}, Iteration: {i:4d} / {len(train_loader):4d}, Loss: {loss.item()}")
        iter_idx += 1


if __name__ == '__main__':
    args, args_text = get_args()
    init_directory(args, args_text)
    args.port = random.randint(10000, 20000)
    logger.info(args_text)

    if args.world_size == 1:
        train(args=args)
    else:
        mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))