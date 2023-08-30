#!/bin/bash
#SBATCH --job-name=raytune
#SBATCH --partition=long
#SBATCH --gres=gpu:2           
#SBATCH --cpus-per-task=8           
#SBATCH --mem=128GB  
#SBATCH --time=2-00:00:00  

module load python/3.9 cuda/11.7 
source ~/venvs/gflownet/bin/activate
python raytune.py