#!/bin/bash

# Create the bash directory if it doesn't exist
bash_dir=logs/bash_training_objectives
mkdir -p $bash_dir

# Loop to create scripts with idx from 0 to 16
# Training objectives has 16 runs
#for idx in {0..15}; do
for idx in {0..15}; do
  # Create the bash script file
  script_name="${bash_dir}/training_objectives_${idx}.sh"
  echo "#!/bin/bash" > $script_name
  echo "#SBATCH --job-name=train_obj_${idx}" >> $script_name
  echo "#SBATCH --partition=long" >> $script_name
  echo "#SBATCH --gres=gpu:1" >> $script_name
  echo "#SBATCH --cpus-per-task=8" >> $script_name
  echo "#SBATCH --mem=64GB" >> $script_name
  #echo "#SBATCH --time=2-00:00:00" >> $script_name
  echo "" >> $script_name
  echo "module load python/3.9 cuda/11.7" >> $script_name
  echo "source ~/venvs/gflownet/bin/activate" >> $script_name
  echo "python raytune.py --experiment_name training_objectives --idx $idx" >> $script_name
  
  # Make the script executable
  chmod +x $script_name
  
  # Run the script
  sbatch $script_name
done

