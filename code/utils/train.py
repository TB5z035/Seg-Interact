import logging

import MinkowskiEngine as ME
import torch
from torch.utils.data import DataLoader

from ..dataset import DATASETS
from .metrics import IoU, mIoU
from ..network import NETWORKS
from .args import get_args
from .misc import get_device, init_directory, init_logger, to_device
from .validate import validate

device = get_device()
logger = logging.getLogger('train')


def train(args):
    # Dataset
    train_dataset = DATASETS[args.train_dataset](args.train_dataset_root, split='train', transform=args.train_transform)
    val_dataset = DATASETS[args.val_dataset](args.val_dataset_root, split='val', transform=args.val_transform)
    assert train_dataset.num_channel == val_dataset.num_channel
    assert train_dataset.num_train_classes == val_dataset.num_train_classes

    # DataLoader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  num_workers=args.train_num_workers,
                                  collate_fn=train_dataset._collate_fn)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                shuffle=False,
                                num_workers=args.val_num_workers,
                                collate_fn=val_dataset._collate_fn)
    # Model
    network = NETWORKS[args.model](train_dataset.num_channel, train_dataset.num_train_classes)
    network = network.to(device)
    # Load pretrained model
    # TODO: Implement this

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
    init_logger(args)
    train(args)
