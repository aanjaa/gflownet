import wandb
from gflownet.tasks.seh_frag import main
import pprint
import torch
from gflownet.tasks.seh_frag import SEHFragTrainer
import os
import shutil

# Create a sweep
count = 5  # Number of hyperparameter combinations to try
sweep_name = "sweep_seh_frag"
sweep_config = {
    'method': 'random'
    }

search_space= {
    "learning_rate": {'values': [3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5]},
    "lr_decay": {'values': [20_000,10_000,1_000]},
    "Z_learning_rate": {'values': [3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4]},
    "Z_lr_decay": {'values': [100_000,50_000,20_000,1_000]},
    }

metric = {
    'name': 'val_loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
sweep_config['parameters'] = search_space

pprint.pprint(sweep_config)

def train():
    training_objective = "TB" #"TB" "FM" "SubTB"
    replay_use = False


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

    hps = {
    "log_dir": "./logs/sweeep_seh_frag",
    "experiment_name": "sweep_seh_frag",
    "device": "cpu",#"cuda" if torch.cuda.is_available() else "cpu",
    "seed": 0, # TODO: how is seed handled?
    "validate_every": 1000,
    "print_every": 100,
    "num_training_steps": 10_000,
    "num_workers": 1,
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
            "Z_learning_rate": 1e-4, #Z_learning_rate,
            "Z_lr_decay": 50_000, #Z_lr_decay,
            "do_parameterize_p_b": False, ### TODO: should be true?
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
        "learning_rate": 1e-3, #learning_rate,
        "lr_decay": 20_000,  #lr_decay,
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

    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHFragTrainer(hps)
    trial.print_every = 5
    info_val = trial.run()

sweep_id = wandb.sweep(sweep_config, project=sweep_name)

wandb.agent(sweep_id, train, count=count)


