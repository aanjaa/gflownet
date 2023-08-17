#!/bin/bash
#SBATCH --job-name=raytune_simple1
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=8                    # Ask for 2 CPUs
#SBATCH --gres=gpu:2                         # Ask for 1 GPU
#SBATCH --mem=32G                             # Ask for 10 GB of RAM

# Loading your environment
source ~/venvs/gflownet/bin/activate

python raytune.py