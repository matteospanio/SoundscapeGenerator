#!/bin/bash
#SBATCH --job-name train_riffusion_model
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user spanio@dei.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 10G
#SBATCH --gres=gpu:rtx:2

# description: Slurm job to train the riffusion model with emotion tags
# author: Mehmet Sanisoglu
# updated: Matteo Spanio

source "$HOME"/miniconda3/bin/activate my_env

WORKDIR="$HOME"/jobs/SoundscapeGenerator
cd "$WORKDIR" || exit 0  # Create and change to the specified directory


export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

srun python train_model.py