#!/bin/bash

#SBATCH --job-name predict
#SBATCH --partition gpu_devel
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus 1
#SBATCH --time 00:30:00 # hh:mm:ss, walltime 
#SBATCH --mem 26000
#SBATCH --output outputs/predict-%j.out
#SBATCH --error outputs/predict-%j.err

python -u src/predict.py "$@"