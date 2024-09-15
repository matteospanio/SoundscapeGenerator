#!/bin/bash
#SBATCH --job-name soundscape_generator
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user spanio@dei.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 16G
#SBATCH --gres=gpu:rtx

# description: Slurm job to generate and categorize the spectrograms
# author: Mehmet Sanisoglu

source $HOME/miniconda3/bin/activate my_env

WORKDIR=$HOME/jobs/SoundscapeGenerator
cd "$WORKDIR" || exit 0  # Create and change to the specified directory

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

mkdir -p log

srun python preparation.py
srun python create_spectrograms.py
srun python prepare_dataset.py
srun python categorize_spectrograms.py
