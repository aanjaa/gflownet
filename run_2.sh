#!/bin/bash
#SBATCH --job-name=run2
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=4           
#SBATCH --mem=32G  

# Loading your environment
source ~/venvs/gflownet/bin/activate

python raytune.py