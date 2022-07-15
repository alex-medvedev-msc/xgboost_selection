#!/bin/bash

#SBATCH --job-name subsample
#SBATCH --partition gpu_devel
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus 1
#SBATCH --time 00:30:00 # hh:mm:ss, walltime 
#SBATCH --mem 26000
#SBATCH --output outputs/subsample-%j.out
#SBATCH --error outputs/subsample-%j.err

python -u src/subsample.py "$@"