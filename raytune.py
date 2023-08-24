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

#Global main 
from gflownet.tasks.main import main

NUM_GPUS = 1
GROUP_FACTORY = tune.PlacementGroupFactory([{'CPU': 2.0, 'GPU': 0.25}])
NUM_WORKERS = 2

FOLDER_NAME = "logs_debug"
NUM_SAMPLES = 16
NUM_TRAINING_STEPS = 10_000 #10_000
VALIDATE_EVERY = 1000 #1000

METRIC = "val_loss"
MODE = "min"   

TASKS = ['seh_frag', 'tdc_frag']             
ORACLES = ['qed','gsk3b','drd2','sa'] 
METHOD_NAMES = ["TB", "FM", "SubTB", "DB"]


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

    tuner = tune.Tuner(
        tune.with_resources(
            functools.partial(main,use_wandb=True),
            resources=GROUP_FACTORY),
        #functools.partial(main,use_wandb=True),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric=METRIC,
            mode=MODE,
            num_samples=NUM_SAMPLES,
            # scheduler=tune.schedulers.ASHAScheduler(grace_period=10),
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
                f"Trial #{i} finished successfully with a {METRIC} metric of: {result.metrics[METRIC]} \n")


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


if __name__ == "__main__":

    # parser = ArgumentParser()
    # parser.add_argument("--experiment_name", type=str,
    #                     default="searchspaces_losses") #["reward_losses", "smoothness_losses", ["searchspaces_losses"], ["replay_and_capacity"], ["exploration_strategies"]][-3]
    # parser.add_argument("--folder", type=str, default="logs_debug")
    # args = parser.parse_args()

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

    ray.init(
        num_cpus=get_num_cpus(), #num_cpus,#8, #num_cpus,
        num_gpus=NUM_GPUS, # num_gpus #2 #num_gpus,
    )

    print(f"Number of cpus: {get_num_cpus()}, number of gpus: {NUM_GPUS}")
    print(f"Placement group factory: {GROUP_FACTORY}")
    print(f"Number of workers: {NUM_WORKERS}")

    config = {
        "log_dir": f"./logs/debug_raytune",
        "device": "cuda" if bool(NUM_GPUS) else "cpu", #"cuda" if torch.cuda.is_available() else "cpu",
        "seed": 0, # TODO: how is seed handled?
        "validate_every": VALIDATE_EVERY,#1000,
        "print_every": 10,
        "num_training_steps": NUM_TRAINING_STEPS,#10_000,
        "num_workers": NUM_WORKERS,
        "num_final_gen_steps": 2, #TODO
        "top_k": 100,
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
            "capacity": 10_000,
            "warmup": 1_000,
            "hindsight_ratio": 0.0,
            "insertion_strategy": "fifo",
            "sampling_strategy": "weighted",
            },
        "cond": {
            "temperature": {
                "sample_dist": "uniform",
                "dist_params": [0, 64.0],
                }
            },
        "task": {
            "name": "seh",
            "tdc": {
                "oracle": "qed"
                }
            },
        }

    learning_rate = tune.choice([3e-4,1e-4,3e-5,1e-5])
    lr_decay = tune.choice([20_000,10_000,1_000])
    Z_learning_rate = tune.choice([3e-2,1e-2,3e-3,1e-3,3e-4,1e-4])
    Z_lr_decay = tune.choice([100_000,50_000,20_000,1_000])

    search_spaces = []
    experiment_name = "training_objectives"

    for task in ['seh_frag']: 
        for training_objective in ["FM"]: #["TB", "FM", "SubTB", "DB"]:

            name = f"{task}_{training_objective}"

            method,method_name,variant = convert_training_obj(training_objective)
            
            replay_use = False

            changes_config = {
                "log_dir": f"./{FOLDER_NAME}/{experiment_name}/{name}",
                "opt.lr_decay": lr_decay,
                "opt.learning_rate": learning_rate,
                "algo.tb.Z_learning_rate": Z_learning_rate,
                "algo.tb.Z_lr_decay": Z_lr_decay,
                "algo.method": method,
                "algo.method_name": method_name,
                "algo.tb.variant": variant,
                "replay.use": replay_use,
                "task.name": task,
                "task.tdc.oracle": "qed"
                }
            
            search_space = change_config(copy.deepcopy(config), changes_config)

            run_raytune(search_space)

            #try:
            #    run_raytune(search_space,metric,num_samples)
            #except:
            #    continue
            
