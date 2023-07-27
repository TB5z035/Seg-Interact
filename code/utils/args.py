import argparse
import os.path as osp

import yaml

from .misc import get_time_str


def get_args():
    parser = argparse.ArgumentParser()

    # Network
    parser.add_argument("--model", type=str, default="MinkUNet34C", help="Model name [default: MinkUNet34C]")
    parser.add_argument("--resume",
                        type=str,
                        default=None,
                        help="Resume training from checkpoint. This is prior to args.pretrained [default: None]")
    parser.add_argument("--pretrained",
                        type=str,
                        default=None,
                        help="Load pretrained model from checkpoint [default: None]")

    # Dataset
    parser.add_argument("--train_dataset",
                        type=dict,
                        help="Dataset config as dictionary for training. See configs/base.yaml")
    parser.add_argument("--val_dataset",
                        type=dict,
                        help="Dataset config as dictionary for validation. See configs/base.yaml")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs [default: 100]")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size during training [default: 1]")
    parser.add_argument("--train_num_workers",
                        type=int,
                        default=4,
                        help="Number of workers for the train loader [default: 4]")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size during validation [default: 1]")
    parser.add_argument("--val_num_workers",
                        type=int,
                        default=4,
                        help="Number of workers for the validation loader [default: 4]")

    parser.add_argument("--labeling_inference",
                        type=bool,
                        default=False,
                        help='whether to update/generate and save pseudo labels')

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
