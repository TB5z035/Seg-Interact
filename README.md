# Interactive Labeling for 2D / 3D Semantic Segmentation

## Environment

See [install.sh](install.sh) for details.

## Training (Fine-tuning)

Training details are specified in config files in the [configs](configs) directory.

```bash
python -m code.utils.train -c [config_path]
```

## Evaluate 

TODO

## Pseudo Labeling

In order to enable active_learning functionalities, the [base_semi.yaml](base_semi.yaml) file under [configs](configs) should be specified as the config path, alongside whether 
to use active_learning and if the inference count should be reset.

```bash
python -m code.utils.train -c [config_path] --labeling_inference --inference_count_reset
```

Within the [base_semi.yaml](base_semi.yaml) file, the following parameters are associated with active learning:

```yaml
world_size: 1                                                                            # currently only supports 1

inf_dataset:
  name: scannet_quantized_fast
  args: 
    root: /hdd1/Guest/tb5zhh/datasets/ScanNet
    split: train
    transform:
      - point_chromatic_normalize:
      - point_to_center:

update_points_num: 20                                                                    # how many ground truth labels are added every inferencing iteration
labeling_inference_epoch: 5                                                              # how many epochs after which inferencing is performed
inference_count_path: '/home/Guest/caiz/labeling_inference/inference_count_1.npy'        # [path to inference count file]
inference_save_path: '/home/Guest/caiz/labeling_inference/run1'                          # [path to inference save folders]
```

After executing the command above, if active learning is specified, the following sequence of codes is performed in one epoch:

[train.py](train.py)
- Inferencing Initialization
  - Counter is reset if specified
  - Previously saved inferencing files are cleared.
- Train
  - Labels are overridden in [scannet.py](scannet.py) if there are updated labels files in [path to inference save folders]
- Validation
- Labeling Inference (only performed every [labeling_inference_epoch])
  - [pseudo_update.py](pseudo_update.py) - Evaluates the model and save prediction point losses, predicted labels, and ground truth labels (previously updated labels as well
    if this is inferencing iteration 0)
  - [point_selection.py](point_selection.py) - Selects the highest losses [update_point_num] points using the previously saved files and saves an updated labels file after swapping
    in [update_point_num] ground truth labels

## Human Labeling (Acquisition Policies)

TODO

## Visualizing
All visualization functionalities are stored under the [vis_and_lab](vis_and_lab) folder and configured with the [base_semi.yaml](base_semi.yaml) file.

### Point Cloud
Note that currently labeling inference must be performed to visualize point clouds. After completing the entire cycle of training, validating, and inferencing, 
.txt files will be generated under the specified "vis_save_path" directory. These files should be used in Meshlab for visualization.

```yaml
vis_save_path: /home/Guest/caiz/labeling_inference/visualize/scannet_scenes              # Path to the saved files
visualize:                                                                               # If visualize, which type(s) should be used?
  - highlight_updated                                                                    # Any of the 3 can be removed
  - color_by_segment
  - color_by_preds
```

There are 3 types of visualizations:
- highlight_updated: the points that were incorporated at every labeling inference will be highlighted whilst other points maintain their true colors
  - [scene_name]_excluded_coords_colors.txt
  - [scene_name]_highlight_updated_coords_colors.txt
<img width="625" alt="image" src="https://github.com/TB5z035/Seg-Interact/assets/98086762/448a4d30-72e1-431d-8791-2a6ac0ffd421">

- color_by_segment: points are colored based on their predicted label and the corresponding RGB value specified in "LABEL_PROTOCAL" ([scannet.py](scannet.py))
  - [scene_name]__pred_seg_coords_colors.txt
  - [scene_name]__gt_seg_coords_colors.txt
<img width="634" alt="image" src="https://github.com/TB5z035/Seg-Interact/assets/98086762/299ba9b8-4e73-4bb3-bd51-66c952f4ce4b">

- color_by_preds: points that are incorrectly labeled will be highlighted (default: white) whereas correctly labeled point will keep their true colors
  - [scene_name]__correct_coords_colors.txt
  - [scene_name]__error_coords_colors.txt
<img width="647" alt="image" src="https://github.com/TB5z035/Seg-Interact/assets/98086762/99a97611-fd7f-49f6-a904-33505b7c1f8f">

### Image
TODO


