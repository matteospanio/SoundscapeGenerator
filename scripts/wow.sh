#!/bin/bash
#SBATCH --job-name wow
#SBATCH --output log/%j_out.txt
#SBATCH --error log/%j_err.txt
#SBATCH --mail-user spanio@dei.unipd.it
#SBATCH --mail-type ALL
#SBATCH --time 2-20:00:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 16G
#SBATCH --gres=gpu:rtx

rm -r /ext/"$USER"
mkdir -p /ext/"$USER"/caches
