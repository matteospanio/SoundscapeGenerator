#!/usr/bin/env bash

# install conda deps
conda install -n my_env \
    wandb pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -c conda-forge -y

# install pip deps
conda run -n my_env \
    pip install -r requirements.txt