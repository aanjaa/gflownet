#!/bin/bash
#SBATCH --job-name=raytune
#SBATCH --partition=unkillable
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10GB
#SBATCH --gpus-per-task=1

# Loading your environment
source ~/venvs/gflownet/bin/activate

python raytune.py