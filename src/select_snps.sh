#!/bin/bash

#SBATCH --job-name select
#SBATCH --partition gpu_devel
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus 1
#SBATCH --time 00:02:00 # hh:mm:ss, walltime 
#SBATCH --mem 8000

python -u src/select_snps.py "$@"