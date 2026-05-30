# ADWM

## Installation

The code is tested with **Python 3.10**, **PyTorch 2.7.1**, and **CUDA 12.8**.

```bash
conda create -n ADWM python=3.10 -y
conda activate ADWM

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Pretrained vision-language backbone

ADWM uses [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) as its vision-language backbone. Download the weights, then set the path to them in `configs/robotwin.yaml` (see [Configuration](#configuration) below).

### Simulation environment

Evaluation requires the **RobotWin2.0** simulation environment. Follow the official setup instructions at:

 https://robotwin-platform.github.io/

---

## Configuration

Paths and hyperparameters are defined in `configs/robotwin.yaml`. Before training or evaluation, edit this file to set at least:

- **`model.vlm.checkpoint_path`** — path to the downloaded Qwen3-VL-2B-Instruct weights.
- **`dataset.dataset_dir`** — path to your (preprocessed) RobotWin2.0 dataset.

---

## Data preprocessing

RobotWin2.0 trajectories often contain near-stationary segments where the robot barely moves. Run the preprocessing script to strip these segments out of the HDF5 data **before training**:

```bash
python clean_dataset_stationary.py
```

---

## Training
```bash
python train_adwm.py 
```
### Useful arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--config` | `./configs/robotwin.yaml` | Path to the config file. |
| `--norm_stats_path` | `./utils/stat-200-10.json` | Path to the action/state normalization statistics. |
| `--save_dir` | `./checkpoints_vla` | Root directory for saving checkpoints. |
| `--grad_accum_steps` | `1` | Gradient accumulation steps. |
| `--save_interval` | `5` | Save a checkpoint every N epochs. |
| `--use_ig` | `True` | Enable the Internal Guidance (IG). |
| `--ig_layer` | `4` | Layer index for IG. |
| `--ig_lambda` | `0.3` | Weight of the Internal Guidance loss. |
| `--use_next_chunk_pred` | `True` | Enable the next-chunk prediction loss. |
| `--lambda_next_chunk` | `0.5` | Weight of the next-chunk prediction loss. |

---

## Evaluation

```bash
python eval_adwm.py
```

---

## Project structure

```
.
├── train_adwm.py                 # Training 
├── eval_adwm.py                  # Inference
├── clean_dataset_stationary.py   # Removes near-stationary segments from RobotWin2.0 HDF5 data
├── configs/
│   └── robotwin.yaml             # Main config: paths, model, dataset, training
├── hdf5_dataloader/              # Dataset and dataloader for RobotWin2.0 HDF5 data
├── models/                       # Model factory, VLA wrapper, action head
└── utils/                        # Normalization stats, VLM/training utilities
```

---


## Acknowledgements

This work builds on the [RobotWin2.0](https://robotwin-platform.github.io/) simulation benchmark and the [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) vision-language model.

