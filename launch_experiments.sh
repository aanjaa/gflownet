#!/bin/bash
#SBATCH --partition=long                      # Ask for unkillable job
#SBATCH --cpus-per-task=10                    # Ask for 2 CPUs
#SBATCH --gres=gpu:4                         # Ask for 1 GPU
#SBATCH --mem=128G                             # Ask for 10 GB of RAM
#SBATCH --time=5:00:00                        # The job will run for 3 hours
#SBATCH -o /network/scratch/j/jarrid.rector-brooks/logs/gfn_exploration-%j.out  # Write the log on tmp1

source /home/mila/j/jarrid.rector-brooks/venvs/gfn_exploration/bin/activate
cd /home/mila/j/jarrid.rector-brooks/repos/gflownet_based_exploration/gfn_exploration

python launch_experiment.py -c 64_64_r0_e2_explore_all_seed -n 16000