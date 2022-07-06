#!/bin/bash

#SBATCH --job-name train_predict
#SBATCH --partition gpu_devel
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --gpus 1
#SBATCH --time 00:30:00 # hh:mm:ss, walltime 
#SBATCH --mem 26000
#SBATCH --output outputs/train_predict-%j.out
#SBATCH --error outputs/train_predict-%j.err

python -u src/train_and_predict.py "$@"