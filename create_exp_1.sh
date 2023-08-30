#!/bin/bash

#Create bash scripts that will run Eperiment 1: Training objevctives

# Define the lists of training_methods and tasks
training_method_list=("method1" "method2" "method3")
task_list=("taskA" "taskB" "taskC")

# Loop through each training_method and task to create Bash files
for training_method in "${training_method_list[@]}"; do
  for task in "${task_list[@]}"; do
    # Create the file name
    file_name="${training_method}_${task}.sh"

    # Create the file and write to it
    echo "#!/bin/bash" > $file_name
    echo "#SBATCH --partition:unkilled" >> $file_name
    echo "python raytune.py --training_method $training_method --task $task" >> $file_name

    # Make the file executable
    chmod +x $file_name
  done
done
