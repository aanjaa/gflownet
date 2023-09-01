#!/bin/bash
experiment_num=0 # change the number of experiment you want to run
prepend_name="onek_" #prepend to the name of experiment
num_random_samples=4 #how many times to sample from shared search space
num_gpus=2 #2 
num_cpus=32 #16 
mem=32 #128

experiment_names_list=("training_objectives" "exploration")
num_trials_list=(16 20)
experiment_name="${experiment_names_list[$experiment_num]}"
num_trials="${num_trials_list[$experiment_num]}"

# Create the bash directory if it doesn't exist
bash_dir="logs/bash/${prepend_name}${experiment_name}"
mkdir -p $bash_dir

# Loop to create scripts
for ((idx=0; idx<$num_trials; idx++)); do
  # Create the bash script file
  script_name="${bash_dir}/${experiment_name}_${idx}.sh"
  echo "#!/bin/bash" > $script_name
  echo "#SBATCH --job-name=${experiment_name}_${idx}" >> $script_name
  echo "#SBATCH --partition=long" >> $script_name
  echo "#SBATCH --gres=gpu:$num_gpus" >> $script_name
  echo "#SBATCH --cpus-per-task=$num_cpus" >> $script_name
  echo "#SBATCH --mem=${mem}GB" >> $script_name
  #echo "#SBATCH --time=2-00:00:00" >> $script_name
  echo "" >> $script_name
  echo "module load python/3.9 cuda/11.7" >> $script_name
  echo "source ~/venvs/gflownet/bin/activate" >> $script_name
  echo "python raytune.py --experiment_name ${experiment_name} --idx $idx --prepend_name ${prepend_name} --num_cpus $num_cpus --num_gpus $num_gpus --num_samples $num_random_samples" >> $script_name
  
  # Make the script executable
  chmod +x $script_name
  
  # Run the script
  sbatch $script_name
done

