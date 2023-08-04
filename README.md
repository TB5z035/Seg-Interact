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
world_size: 1                                                                            #currently only supports 1

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
- Inferencing Init
  - Counter is reset if specified
  - Previously saved inferencing files are cleared.
- Train
  - Labels are overridden in [scannet.py](scannet.py) if there are updated labels files in [path to inference save folders]
- Validation
- Labeling Inference (only performed every [labeling_inference_epoch])
  - [pseudo_update.py](pseudo_update.py)
    - Evaluates the model and save prediction point losses, predicted labels, and ground truth labels (previous updated labels as well
      if this is inferencing iteration 0)
  - [point_selection.py](point_selection.py)
    - Selects the highest losses [update_point_num] points using the previously saved files and saves an updated labels file after swapping
      in [update_point_num] ground truth labels

## Human Labeling (Acquisition Policies)

TODO

