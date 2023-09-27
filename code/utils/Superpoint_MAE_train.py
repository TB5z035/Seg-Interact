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

from ..optimizer import OPTIMIZERS, SCHEDULERS
from .args import get_args
from .misc import get_device, init_directory, init_logger, to_device, get_local_rank, get_world_size, save_checkpoint
from .validate import validate

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
    if hasattr(args, 'datamodule'):
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("eval", eval)
        sp_dataset = DATASETS[args.datamodule['sp_base']](**(args.train_dataset['args'] | {'sp_cfg': args.datamodule}))

        train_dataset = DATASETS[args.train_dataset['name']](**(args.train_dataset['args'] | {'sp_cls': sp_dataset}))
        val_dataset = DATASETS[args.val_dataset['name']](**args.val_dataset['args'] | {'sp_cls': sp_dataset})
    else:
        train_dataset = DATASETS[args.train_dataset['name']](**(args.train_dataset['args']))
        val_dataset = DATASETS[args.val_dataset['name']](**args.val_dataset['args'])

    assert train_dataset.num_channel == val_dataset.num_channel
    assert train_dataset.num_train_classes == val_dataset.num_train_classes
    logger.info(f"Train dataset: {args.train_dataset}\nVal dataset: {args.val_dataset}")

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
        shuffle=False,  # (train_sampler is None),
        num_workers=args.train_num_workers,
        sampler=train_sampler,
        collate_fn=train_dataset._collate_fn if hasattr(train_dataset, '_collate_fn') else None)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                shuffle=False,
                                num_workers=args.val_num_workers,
                                collate_fn=val_dataset._collate_fn)

    logger.info(f"Train dataloader: {len(train_dataloader)}\nVal dataloader: {len(val_dataloader)}")

    # Model
    # print(train_dataset.num_channel, train_dataset.num_train_classes)
    network = NETWORKS[args.model['name']](args.model['args'])
    network = network.to(device)

    for _, data in enumerate(train_dataloader):
        inputs, labels, extras = data[0], data[1], data[2]
        rec_x, rec_x_indices = network(inputs, extras)
        print(rec_x.shape, rec_x_indices.shape)
        exit()

    # for index, data in enumerate(train_dataloader):
    '''
    batch size = b
    number of points in scene = N
    number of superpoints = x

    data[0]: (coords, colors)
        coords: torch.tensor(b*N, 4) -> [(batch_index, x, y, z), ...]
        colors: torch.tensor(b*N, 3) -> [(r, g, b), ...]

    data[1]: labels
        labels: torch.tensor(b*N, )

    data[2]: extras -> {'scene_id': tuple(b*N, ),
                        'full_super_indices': tuple(torch.tensor(N, ), ...),
                        'full_features': tuple(torch.tensor(N, 10), ...) -> (x, y, z, r, g, b, lin, pla, sca, ele)}

    '''

    # (coords, colors), labels, extras = data
    # # print(coords.shape)
    # print(full_features.shape)
    # print(len(extras['superpoint_sizes'][0]))
    # print(coords[:, 0])

    # exit()

    # Load pretrained model
    # if args.resume:
    #     logger.info(f"Resume training from {args.resume}")
    #     ckpt = torch.load(args.resume, map_location=device)
    #     network.load_state_dict(ckpt['network'])
    # elif args.model['args']['pretrained']:
    #     logger.info(f"Load pretrained model from {args.pretrained}")
    #     ckpt = torch.load(args.model['args']['pretrained'], map_location=device)
    #     network.load_state_dict(ckpt['network'])
    #     pass
    # network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
    logger.info(f"Model: {args.model}")

    # Optimizer
    optimizer = OPTIMIZERS[args.optimizer['name']](network.parameters(), **args.optimizer['args'])
    '''
    if args.resume:
        optimizer.load_state_dict(ckpt['optimizer'])

    # Criterion
    # TODO: init from args
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

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
    
    '''

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

        if epoch_idx % args.save_epoch_freq == 0:
            save_checkpoint(network, args, epoch_idx, global_iter[0], optimizer, scheduler, name=f'epoch#{epoch_idx}')


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
        output = model(inputs, extras)
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


if __name__ == '__main__':
    args, args_text = get_args()
    init_directory(args, args_text)
    args.port = random.randint(10000, 20000)
    logger.info(args_text)

    if args.world_size == 1:
        train(args=args)
    else:
        mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))
