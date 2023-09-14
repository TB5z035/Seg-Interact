import argparse
import os.path as osp
import yaml
from .misc import get_time_str

import pyrootutils
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


def init_spconfig(config_name: str, overrides=[]):
    assert config_name is not None, 'Using superpoint but config file name not specified'
    GlobalHydra.instance().clear()
    pyrootutils.setup_root(".", pythonpath=True)
    with initialize(version_base='1.2', config_path="../../sp_configs"):
        sp_cfg = compose(config_name=config_name, overrides=overrides)
    return sp_cfg

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

    # Inference
    parser.add_argument("--labeling_inference",
                        default=False,
                        action='store_true',
                        help='whether to perform labeling update inference')
    parser.add_argument("--inference_count_reset",
                        default=False,
                        action='store_true',
                        help='whether to reset inference count')
    parser.add_argument("--inference_count_path", type=str, default='', help='path to inference count file')
    parser.add_argument("--update_points_num",
                        type=int,
                        default=200,
                        help='number of points to update after each inference')
    parser.add_argument("--labeling_inference_epoch", type=int, default=5)
    parser.add_argument("--inference_save_path", type=str, default='')

    # Visualization
    parser.add_argument("--visualize", type=list, default=None, help='whether or how to visualize point cloud')
    parser.add_argument("--vis_save_path",
                        type=str,
                        default='/home/Guest/caiz/labeling_inference/visualize/scannet_scenes1')

    # Do we have a config file to parse?
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c',
                               '--config',
                               default=None,
                               type=str,
                               help='YAML config file specifying default arguments')

    config_parser.add_argument('-s',
                                '--spconfig',
                                default=None,
                                type=str,
                                help='YAML config file specifying superpoint arguments')
    
    args_config, remaining = config_parser.parse_known_args()
    assert args_config.config is not None, 'Config file must be specified'

    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    
    if args_config.spconfig is not None:
        sp_cfg = init_spconfig(config_name=osp.split(args_config.spconfig)[1])
        parser.set_defaults(**sp_cfg)

    args = parser.parse_args(remaining)
    args.exp_dir = osp.join('experiments', osp.relpath(args_config.config, 'configs')[:-5])
    args.start_time = get_time_str()
    if args.labeling_inference:
        assert args.inference_count_path is not None, 'inference count path not specified'
        assert args.update_points_num != 0, 'update points num = 0, not point in performing labeling inference'
    if args.visualize:
        assert args.labeling_inference, 'labeling inference must be performed in order to use visualization functions'
    if args.resume:
        resume_path = args.resume
        import torch
        cfg = yaml.safe_load(torch.load(args.resume, map_location='cpu')['args'])
        parser.set_defaults(**cfg)
        args = parser.parse_args(remaining)
        args.resume = resume_path

    # Cache the args as a text string to save them in the output dir later
    args_dict = args.__dict__.copy()
    remove_keys = ['datamodule', 'extras', 'local', 'paths']
    for key in remove_keys:
        args_dict.pop(key, None)

    args_text = yaml.safe_dump(args_dict, default_flow_style=False)
    del args_dict
    
    return args, args_text


if __name__ == '__main__':
    args, args_text = get_args()
    print(args)
    print(args_text)
    print(args.exp_dir)
