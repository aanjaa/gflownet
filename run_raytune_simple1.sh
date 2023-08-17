#!/bin/bash
#SBATCH --job-name=raytune_simple1
#SBATCH --partition=unkillable                      # Ask for unkillable job
#SBATCH --cpus-per-task=6                    # Ask for 2 CPUs
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G                             # Ask for 10 GB of RAM

# Loading your environment
source ~/venvs/gflownet/bin/activate

python raytune.py