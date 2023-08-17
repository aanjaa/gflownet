#!/bin/bash
#SBATCH --job-name=raytune_simple1
#SBATCH --partition=main
#SBATCH --gres=gpu:2                
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                    

# Loading your environment
source ~/venvs/gflownet/bin/activate

python raytune.py