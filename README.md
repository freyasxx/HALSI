# HALSI: Hierarchical Adaptive Latent Skill Inference for Temporal Abstraction in Reinforcement Learning

This project provides the official implementation of HALSI, a hierarchical reinforcement learning (HRL) framework designed to enable dynamic latent skill abstraction and adaptive temporal decision-making. HALSI combines a Transformer-based skill encoder, a diffusion-based skill decoder, and an adaptive duration policy, allowing agents to learn semantically meaningful skills from offline data and execute them with variable temporal horizons during online training.

Our implementation is partially based on [ReSkill](https://arxiv.org/abs/2211.02231), particularly for the environment setup. We thank the authors for open-sourcing their codebase, which served as a valuable reference for this project.

The framework supports:

Pretraining a skill representation module from demonstration trajectories.

Hierarchical policy optimization, integrating high-level skill modulation and low-level soft blending for fine-grained control.

Experiments on manipulation tasks in Fetch environments (e.g., block manipulation, hook manipulation) and integration with offline datasets.

## Directory Structure

```
├── data_p.py                # Data processing script
├── environment.yml          # Conda environment configuration
├── README.md                # Project documentation
├── setup.py                 # Installation script
├── dataset/                 # Collected demonstration data
│   └── fetch_block_40000/
│       └── demos.npy
├── halsi/                   # Core project code
│   ├── train_HRL_agent.py   # HRL agent training script
│   ├── train_skill_modules.py # Skill module training script
│   ├── configs/             # Configuration files
│   ├── data/                # Data-related code
│   ├── models/              # Model code
│   ├── results/             # Results and logs
│   ├── rl/                  # Reinforcement learning code
│   └── utils/               # Utility functions
├── results/                 # Training results
└── saved_dskill_models/     # Saved skill models
```

## Environment Setup

It is recommended to use a Conda environment:

```sh
conda env create -f environment.yml
conda activate <your_env_name>
```

## Dataset

Run `halsi/data/collect_demos.py` to collect demonstration data, or download preprocessed datasets from [here](https://drive.google.com/drive/folders/1yTr_6fc-sHXK_CZkm8QIRTV9VgWxKpOE) and place the extracted `dataset` folder in the project root.

## Train Skill Modules

```sh
python halsi/train_skill_modules.py --config_file block/config.yaml --dataset_name fetch_block_40000
```

## Test Skill Modules

```sh
python halsi/utils/test_skill_modules.py --dataset_name fetch_block_40000 --task block --use_skill_prior True
```

## Train ReSkill Agent

```sh
python halsi/train_HRL_agent.py --config_file table_cleanup/config.yaml --dataset_name fetch_block_40000
```

## Logging

All results are logged using [Weights and Biases](https://wandb.ai). Please register and initialize your account before first use.

## Code Structure

- `dataset/`: Collected demonstration data
- `halsi/`: Core code including data collection, model training, utilities
- `results/`: Training results and logs
- `saved_dskill_models/`: Saved skill models

