import copy
import functools
import json
import os
import shutil

from ray import air, tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from pathlib import Path

from argparse import ArgumentParser
import time
import torch
import ray

_SLURM_JOB_CPUS_FILENAME = '/sys/fs/cgroup/cpuset/slurm/uid_%s/job_%s/cpuset.cpus'

def replace_dict_key(dictionary, key, value):
    keys = key.split(".")
    current_dict = dictionary

    for k in keys[:-1]:
        if k in current_dict:
            current_dict = current_dict[k]
        else:
            raise KeyError(f"Key '{key}' does not exist in the dictionary.")

    last_key = keys[-1]
    if last_key in current_dict:
        current_dict[last_key] = value
    else:
        raise KeyError(f"Key '{key}' does not exist in the dictionary.")

    return dictionary

def convert_str_to_bool(args_obj, args_to_convert):
    for arg in args_to_convert:
        # check format
        curr_value = getattr(args_obj, arg)
        try:
            assert curr_value in ['true', 'false']
        except:
            raise ValueError(f' Argument {arg}={curr_value} needs to provided as true or false')
        # set bool format
        bool_value = True if curr_value == 'true' else False
        setattr(args_obj, arg, bool_value)
    return args_obj

def change_config(config,changes_config):
    for key, value in changes_config.items():
        config = replace_dict_key(config, key, value)
    return config

def get_num_cpus() -> int:
    '''
    In a typical slurm allocation we only have access to a subset of the
    current node's CPUs.  Allocating ray with more CPUs than we actually
    have naturally leads to massive slowdowns in code performance, so
    we should avoid this.

    For any job allocation, Slurm writes a file to /sys listing the
    CPUs on the node which belong to the allocation.  If the code is
    being run on a slurm allocation this function reads the slurm CPU file
    and returns the number of CPUs allocated for the job.  If not on a slurm
    allocation, ray is initialized by default with the number of CPUs available
    on the system.
    '''
    if 'SLURM_JOB_ID' not in os.environ:
        return os.cpu_count()

    uid, slurm_job_id = os.getuid(), os.environ['SLURM_JOB_ID']
    fname = Path(_SLURM_JOB_CPUS_FILENAME % (uid, slurm_job_id))

    num_cpus = 0
    with open(fname, 'r') as f:
        line = f.read().replace('\n', '')
        for substr in line.split(','):
            if '-' not in substr:
                num_cpus += 1
                continue

            cpu_nums = list(map(int, substr.split('-')))
            num_cpus += cpu_nums[1] - cpu_nums[0] + 1

    return num_cpus

def run_raytune(main,search_space,metric,num_samples,experiment_name,name):

    if os.path.exists(search_space["log_dir"]):
        if search_space["overwrite_existing_exp"]:
            shutil.rmtree(search_space["log_dir"])
        else:
            raise ValueError(f"Log dir {search_space['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
        
    os.makedirs(search_space["log_dir"])

    # Save the search space
    with open(os.path.join(search_space["log_dir"] + "/" + time.strftime("%d.%m_%H:%M:%S") + ".json"), 'w') as fp:
        json.dump(search_space, fp, sort_keys=True, indent=4, skipkeys=True,
                    default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")

    group_factory = tune.PlacementGroupFactory([
        {'CPU': CPU, 'GPU': CPU} #for _ in range(2)
    ])

    # Save the search space by saving this file itself
    shutil.copy(__file__, os.path.join(search_space["log_dir"] + "/ray.py"))

    #group_factory = get_placement_group_factory()
    print(group_factory)
    tuner = tune.Tuner(
        tune.with_resources(
            functools.partial(main,use_wandb=True),
            resources=group_factory),
        #main,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode="min",
            num_samples=num_samples,
            # scheduler=tune.schedulers.ASHAScheduler(grace_period=10),
            search_alg=BasicVariantGenerator(constant_grid_search=True),
            # search_alg=OptunaSearch(mode="min", metric="valid_loss_outer"),
            # search_alg=Repeater(OptunaSearch(mode="min", metric="valid_loss_outer"), repeat=2),
        ),
        run_config=air.RunConfig(name="details", verbose=1,local_dir=search_space["log_dir"], log_to_file=False)
    )
    
    #ray.init()
    
    # Start timing 
    start = time.time()

    results = tuner.fit()

    # Stop timing
    end = time.time()
    print(f"Time elapsed: {end - start}")

    # Get a DataFrame with the results and save it to a CSV file
    df = results.get_dataframe()
    df.to_csv(os.path.join(search_space["log_dir"] + "/" + 'dataframe.csv'), index=False)

    # Generate txt files
    if results.errors:
        print("ERROR!")
    else:
        print("No errors!")
    if results.errors:
        with open(os.path.join(search_space["log_dir"], "error.txt"), 'w') as file:
            file.write(f"Experiment failed for with errors {results.errors}")

    with open(os.path.join(search_space["log_dir"] + "/summary.txt"), 'w') as file:
        for i, result in enumerate(results):
            if result.error:
                file.write(f"Trial #{i} had an error: {result.error} \n")
                continue

            file.write(
                f"Trial #{i} finished successfully with a {metric} metric of: {result.metrics[metric]} \n")


    config = results.get_best_result().config
    with open(os.path.join(search_space["log_dir"] + "/best_config.json"), 'w') as file:
        json.dump(config, file, sort_keys=True, indent=4, skipkeys=True,
                    default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")


if __name__ == "__main__":

    # parser = ArgumentParser()
    # parser.add_argument("--experiment_name", type=str,
    #                     default="searchspaces_losses") #["reward_losses", "smoothness_losses", ["searchspaces_losses"], ["replay_and_capacity"], ["exploration_strategies"]][-3]
    # parser.add_argument("--folder", type=str, default="logs_debug")
    # args = parser.parse_args()

    #folder_name = args.folder
    num_cpus = get_num_cpus()
    num_gpus = torch.cuda.device_count()

    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )

    CPU = num_cpus
    GPU = float(torch.cuda.is_available())
    num_workers = num_cpus
    num_samples = 2

    TASKS = ['seh', 'jnk3', 'gsk3b', 'celecoxib_rediscovery',
    'troglitazone_rediscovery',
    'thiothixene_rediscovery', 'albuterol_similarity', 'mestranol_similarity',
    'isomers_c7h8n2o2', 'isomers_c9h10n2o2pf2cl', 'median1', 'median2', 'osimertinib_mpo',
    'fexofenadine_mpo', 'ranolazine_mpo', 'perindopril_mpo', 'amlodipine_mpo',
    'sitagliptin_mpo', 'zaleplon_mpo', 'valsartan_smarts', 'deco_hop', 'scaffold_hop', 'qed', 'drd2']
    
    TRAINING_OBJECTIVES = ["TB", "FM", "SubTB"]

    #training_objective = "TB"
    #task = "seh"

    #experiment_name = "seh_compare_training_objectives"
    #name = training_objective


    metric = "val_loss"


    config = {
        "log_dir": f"./logs/raytune",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0, # TODO: how is seed handled?
        "validate_every": 1,#1000,
        "print_every": 1,
        "num_training_steps": 1,#10_000,
        "num_workers": num_workers,
        "overwrite_existing_exp": True,
        "algo": {
            "method": "TB",
            "sampling_tau": 0.9,
            "global_batch_size": 64,
            "offline_ratio": 0.0,
            "valid_offline_ratio": 0.0,
            "max_nodes": 9,
            "illegal_action_logreward": -75,
            "train_random_action_prob": 0.0,
            "valid_random_action_prob": 0.0,
            "tb": {
                "do_subtb": False,
                "Z_learning_rate": 1e-3,
                "Z_lr_decay": 50_000,
                "do_parameterize_p_b": False,
                "epsilon": None,
                "bootstrap_own_reward": False,
                },
            },
        "model": {
            "num_layers": 4,
            "num_emb": 128,
            },
        "opt": {
            "opt": "adam",
            "learning_rate": 1e-4,
            "lr_decay": 20_000,
            "weight_decay": 1e-8,
            "momentum": 0.9,
            "clip_grad_type": "norm",
            "clip_grad_param": 10,
            "adam_eps": 1e-8,
            },
        "replay": {
            "use": False,
            "capacity": 10_000,
            "warmup": 1_000,
            "hindsight_ratio": 0.0,
            },
        "cond": {
            "temperature": {
                "sample_dist": "uniform",
                "dist_params": [0, 64.0],
                }
            },
        "task": {
            "tdc": {
                "oracle_name": "qed",
                }
            },
        }
    
    def convert_task_and_training_obj(task,training_objective):
        if task == "seh":
            from gflownet.tasks.seh_frag import main
            oracle_name = "" #dummy
        else:
            from gflownet.tasks.tdc_opt import main
            oracle_name = task


        if training_objective == "TB":
            method = "TB"
            do_subtb = False
        elif training_objective == "FM":
            method = "FM"
            do_subtb = False
        elif training_objective == "SubTB":
            method = "TB"
            do_subtb = True
        else:
            raise ValueError(f"Training objective {training_objective} not supported")
        return main, task, oracle_name,method, do_subtb


    learning_rate = tune.choice([3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5])
    lr_decay = tune.choice([20_000,10_000,1_000])
    Z_learning_rate = tune.choice([3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4])
    Z_lr_decay = tune.choice([100_000,50_000,20_000,1_000])


    search_spaces = []
    experiment_name = "training_objectives"

    for task in ['seh','albuterol_similarity', 'qed']:
        for training_objective in TRAINING_OBJECTIVES:

            name = f"{task}_{training_objective}"

            main, task, oracle_name,method,do_subtb = convert_task_and_training_obj(task,training_objective)
            
            replay_use = False

            changes_config = {
                "log_dir": f"./logs/{experiment_name}/{name}",
                "opt.lr_decay": lr_decay,
                "opt.learning_rate": learning_rate,
                "algo.tb.Z_learning_rate": Z_learning_rate,
                "algo.tb.Z_lr_decay": Z_lr_decay,
                "algo.method": method,
                "algo.tb.do_subtb": do_subtb,
                "replay.use": replay_use,
                "task.tdc.oracle_name": oracle_name,
                }
            
            search_space = change_config(copy.deepcopy(config), changes_config)
            try:
                run_raytune(main,search_space,metric,num_samples,experiment_name,name)
            except:
                continue
            
