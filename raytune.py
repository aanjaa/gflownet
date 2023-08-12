import copy
import functools
import json
import os

from ray import air, tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from gflownet.tasks.seh_frag import main

from argparse import ArgumentParser
import time
import torch

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



if __name__ == "__main__":

    # parser = ArgumentParser()
    # parser.add_argument("--experiment_name", type=str,
    #                     default="searchspaces_losses") #["reward_losses", "smoothness_losses", ["searchspaces_losses"], ["replay_and_capacity"], ["exploration_strategies"]][-3]
    # parser.add_argument("--folder", type=str, default="logs_debug")
    # args = parser.parse_args()

    #folder_name = args.folder
    training_objective = "TB" #"TB" "FM" "SubTB"
    replay_use = False


    metric = "val_loss"


    if training_objective == "TB":
        method = "TB"
        do_subtb = False
    elif training_objective == "FM":
        method = "FM"
    elif training_objective == "SubTB":
        method = "TB"
        do_subtb = True
    else:
        raise ValueError(f"Training objective {training_objective} not supported")

    num_samples = 1

    method = "TB" 

    learning_rate = tune.choice([3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5])
    Z_learning_rate = tune.choice([3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4])
    Z_lr_decay = tune.choice([100_000,50_000,20_000,1_000,0])
    lr_decay = tune.choice([20_000,10_000,1_000,0])

    search_space = {
    "log_dir": "./logs/debug_run_seh_frag",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0, # TODO: how is seed handled?
    "validate_every": 1000,
    "print_every": 100,
    "num_training_steps": 10_000,
    "num_workers": 5,
    "overwrite_existing_exp": True,
    "algo": {
        "method": method,
        "sampling_tau": 0.9,
        "global_batch_size": 64,
        "offline_ratio": 0.0,
        "valid_offline_ratio": 0.0,
        "max_nodes": 9,
        "illegal_action_logreward": -75,
        "train_random_action_prob": 0.0,
        "valid_random_action_prob": 0.0,
        "tb": {
            "do_subtb": do_subtb,
            "Z_learning_rate": Z_learning_rate,
            "Z_lr_decay": Z_lr_decay,
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
        "learning_rate": learning_rate,
        "lr_decay": lr_decay,
        "weight_decay": 1e-8,
        "momentum": 0.9,
        "clip_grad_type": "norm",
        "clip_grad_param": 10,
        "adam_eps": 1e-8,
        },
    "replay": {
        "use": replay_use,
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
    }

    # Save the search space
    with open(os.path.join(search_space["log_dir"] + "/" + time.strftime("%d.%m_%H:%M:%S") + ".json"), 'w') as fp:
        json.dump(search_space, fp, sort_keys=True, indent=4, skipkeys=True,
                    default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")

    tuner = tune.Tuner(
        #tune.with_resources(functools.partial(main, use_wandb=False), {"cpu": 1.0, "gpu": 1.0}),
        functools.partial(main, use_wandb=False),
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
        run_config=air.RunConfig(name="details", verbose=2,local_dir=search_space["log_dir"], log_to_file=False)
    )

    # Generate txt files
    results = tuner.fit()
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
