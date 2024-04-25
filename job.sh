#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=6:00:00

source ~/gflownet/setup_mila_cluster.sh
cd ~/gflownet/src/gflownet/tasks/
python main.py use_wandb=True seed=$SLURM_ARRAY_TASK_ID  "$@"