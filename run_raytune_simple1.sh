#!/bin/bash
#SBATCH --job-name=raytune_simple1
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=4                    # Ask for 2 CPUs
#SBATCH --gres=gpu:4                         # Ask for 1 GPU
#SBATCH --mem=5G                             # Ask for 10 GB of RAM

# Loading your environment
source ~/venvs/gflownet/bin/activate

python raytune.py