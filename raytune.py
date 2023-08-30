import copy
import functools
import json
import os
import shutil

from ray import air, tune
from ray.tune.search.basic_variant import BasicVariantGenerator
import multiprocessing

from argparse import ArgumentParser
import time
import torch
import ray
from gflownet.utils.misc import replace_dict_key,change_config, get_num_cpus
from gflownet.algo.config import TBVariant
from ray.tune.schedulers import ASHAScheduler

#Global main 
from gflownet.tasks.main import main

def run_raytune(search_space):

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

    # Save the search space by saving this file itself
    shutil.copy(__file__, os.path.join(search_space["log_dir"] + "/ray.py"))

    # asha_scheduler = ASHAScheduler(
    # time_attr='training_iteration',
    # metric='loss',
    # mode='min',
    # max_t=100,
    # grace_period=10,
    # reduction_factor=3,
    # brackets=1,
    # )

    tuner = tune.Tuner(
        tune.with_resources(
            functools.partial(main,use_wandb=True),
            resources=group_factory),
        #functools.partial(main,use_wandb=True),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
            # scheduler=asha_scheduler,
            search_alg=BasicVariantGenerator(constant_grid_search=True),
            # search_alg=OptunaSearch(mode="min", metric="valid_loss_outer"),
            # search_alg=Repeater(OptunaSearch(mode="min", metric="valid_loss_outer"), repeat=2),
        ),
        run_config=air.RunConfig(name="details", verbose=2,local_dir=search_space["log_dir"], log_to_file=False)
    )
    
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


def convert_training_obj(training_objective):
    if training_objective == "FM":
        method = "FM"
        variant = TBVariant.TB
        method_name = "FM"
    elif training_objective == "TB":
        method = "TB"
        variant = TBVariant.TB
        method_name = "TB"
    elif training_objective == "SubTB1":
        method = "TB"
        variant = TBVariant.SubTB1
        method_name = "SubTB1"
    elif training_objective == "DB":
        method = "TB"
        variant = TBVariant.DB
        method_name = "DB"
    else:
        raise ValueError(f"Training objective {training_objective} not supported")
    return method, method_name, variant


def convert_task(task):
    if task == "seh_frag":
        task_name = "seh_frag"
        oracle = "qed"

    elif task  == 'qed_frag':
        task_name = "tdc_frag"
        oracle = "qed"

    elif task == 'gsk3_frag':
        task_name = "tdc_frag"
        oracle = "gsk3"

    elif task == 'drd2_frag':
        task_name = "tdc_frag"
        oracle = "drd2"

    elif task == 'sa_frag':
        task_name = "tdc_frag"
        oracle = "sa"
    else:
        raise ValueError(f"Task {task} not supported")
    return task_name,oracle


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str,
                        default="training_objectives") 
    parser.add_argument("--idx", type=int, default=0, help ="Run number in an experiment") 
    args = parser.parse_args()

    #folder_name = args.folder

    # num_cpus = get_num_cpus()
    # if use_gpus:
    #     if torch.cuda.is_available():
    #         #num_gpus = torch.cuda.device_count() #this doesn't always work on the cluster
    #         num_gpus = num_gpus
    #     else:
    #         print("No GPUs available")
    #         num_gpus = 0
        
    # else:
    #     num_gpus = 0
    # print(f"num_cpus: {num_cpus}, num_gpus: {num_gpus}")

    batch_experiment_name = "trial" #+ time.strftime("%d.%m_%H:%M:%S")
    folder_name = "logs"

    num_gpus = 1
    group_factory = tune.PlacementGroupFactory([{'CPU': 8.0, 'GPU': 1.0}])
    num_workers = 7

    num_samples = 1
    num_training_steps = 15_650 #10_000
    validate_every = 1000 #1000

    metric = "val_loss"
    mode = "min"   

    training_objectives =  ["TB", "FM", "SubTB1", "DB"]
    tasks = ['seh_frag','qed_frag','drd2_frag','sa_frag']

    #tasks = ['seh_frag', 'tdc_frag']             
    #oracles = ['qed','drd2','sa'] 

    ray.init(
        num_cpus=get_num_cpus(), #num_cpus,#8, #num_cpus,
        num_gpus=num_gpus, # num_gpus #2 #num_gpus,
    )

    print(f"Number of cpus: {get_num_cpus()}, number of gpus: {num_gpus}")
    print(f"Placement group factory: {group_factory}")
    print(f"Number of workers: {num_workers}")

    config = {
        "log_dir": f"./logs/debug_raytune",
        "device": "cuda" if bool(num_gpus) else "cpu",
        "seed": 0, # TODO: how is seed handled?
        "validate_every": validate_every,#1000,
        "print_every": 10,
        "num_training_steps": num_training_steps,#10_000,
        "num_workers": num_workers,
        "num_final_gen_steps": 320,
        "overwrite_existing_exp": True,
        "algo": {
            "method": "TB",
            "method_name": "TB",
            "sampling_tau": 0.9,
            "global_batch_size": 64,
            "offline_ratio": 0.0,
            "valid_offline_ratio": 0.0,
            "max_nodes": 9,
            "illegal_action_logreward": -75,
            "train_random_action_prob": 0.0,
            "valid_random_action_prob": 0.0,
            "tb": {
                "variant": TBVariant.TB,
                "Z_learning_rate": 1e-3,
                "Z_lr_decay": 50_000,
                "do_parameterize_p_b": False,
                "do_length_normalize": False, ###TODO
                "epsilon": None,
                "bootstrap_own_reward": False,
                "cum_subtb": True,
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
            "capacity": 100,
            "warmup": 100,
            "hindsight_ratio": 0.0,
            "insertion": {
                "strategy": "fifo",#"diversity_and_reward_fast",
                "sim_thresh": 0.7,
                "reward_thresh": 0.9,
                },
            "sampling":{
                "strategy": "uniform",
                "weighted": {
                    "reward_power": 1.0,
                    },
                "quantile": {
                    "alpha": 0.1,
                    "beta": 0.5,
                    },
                },
            },
        "cond": {
            "temperature": {
                "sample_dist": "constant", #"uniform"
                "dist_params": [1.0],#[0, 64.0],  #[16,32,64,96,128]
                "num_thermometer_dim": 1,
                }
            },
        "task": {
            "name": "seh",
            "tdc": {
                "oracle": "qed"
                }
            },
        "evaluation": {
            "k": 100,
            "reward_thresh": 8.0,
            "tanimoto_thresh": 0.7,
            },
        }

    learning_rate = tune.choice([3e-4,1e-4,3e-5,1e-5])
    lr_decay = tune.choice([20_000,10_000,1_000])
    Z_learning_rate = tune.choice([3e-2,1e-2,3e-3,1e-3,3e-4,1e-4])
    Z_lr_decay = tune.choice([100_000,50_000,20_000,1_000])

    search_spaces = []

    if args.experiment_name == "training_objectives":
        for task in tasks: #["sa_frag"]: #tasks:
            for training_objective in training_objectives:

                name = f"{task}_{training_objective}"

                method,method_name,variant = convert_training_obj(training_objective)
                task_name, oracle = convert_task(task)

                changes_config = {
                    "log_dir": f"./{folder_name}/{args.experiment_name}_{batch_experiment_name}/{name}",
                    "opt.lr_decay": lr_decay,
                    "opt.learning_rate": learning_rate,
                    "algo.tb.Z_learning_rate": Z_learning_rate,
                    "algo.tb.Z_lr_decay": Z_lr_decay,
                    "algo.method": method,
                    "algo.method_name": method_name,
                    "algo.tb.variant": variant,
                    "task.name": task_name,
                    "task.tdc.oracle": oracle,
                    }
                
                search_spaces.append(change_config(copy.deepcopy(config), changes_config)) 

        print(f"Running run number {args.idx} out of {len(search_spaces)}")
        run_raytune(search_spaces[args.idx])

            #try:
            #    run_raytune(search_space,metric,num_samples)
            #except:
            #    continue
            
