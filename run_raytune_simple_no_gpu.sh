#!/bin/bash
#SBATCH --job-name=raytune
#SBATCH --partition=main                    ##unkillable
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=10GB

# Loading your environment
source ~/venvs/gflownet/bin/activate

python raytune.py