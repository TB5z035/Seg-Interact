import argparse
import os.path as osp

import yaml

from .misc import get_time_str

def get_args():
    parser = argparse.ArgumentParser()

    # Network
    parser.add_argument("--model", type=str, default="MinkUNet34C")

    # Dataset
    parser.add_argument("--train_dataset", type=str, default="scannet")
    parser.add_argument("--train_dataset_root", type=str, default="data")
    parser.add_argument("--val_dataset", type=str, default="scannet")
    parser.add_argument("--val_dataset_root", type=str, default="data")
    parser.add_argument("--train_transform", type=object, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--train_num_workers", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--val_num_workers", type=int, default=4)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help='path for pretrained checkpoints')
    parser.add_argument("--labeling_inference", type=bool, default=False, help='whether to update/generate and save pseudo labels')
    # parser.add_argument("--label_update_epoch", type=int, default=2)

    # Do we have a config file to parse?
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c',
                               '--config',
                               default=None,
                               type=str,
                               help='YAML config file specifying default arguments')
    
    args_config, remaining = config_parser.parse_known_args()
    assert args_config.config is not None, 'Config file must be specified'

    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args.exp_dir = osp.join('experiments', osp.relpath(args_config.config, 'configs')[:-5])
    args.start_time = get_time_str()
    if args.resume:
        resume_path = args.resume
        import torch
        cfg = yaml.safe_load(torch.load(args.resume, map_location='cpu')['args'])
        parser.set_defaults(**cfg)
        args = parser.parse_args(remaining)
        args.resume = resume_path

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

if __name__ == '__main__':
    args, args_text = get_args()
    print(args)
    print(args_text)
    print(args.exp_dir)