import os
import shutil
import wandb

import numpy as np
from omegaconf import OmegaConf

from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.algo.config import TBVariant
import torch
import time
import random
import argparse

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def main(hps, use_wandb=False, entity="evaluating-gfns"):
    # hps must contain task.name, log_dir, overwrite_existing_exp
    # Measuring time
    seed_everything(hps["seed"])
    start_time = time.time()

    if use_wandb:
        wandb.init(
            entity=entity, project=hps["log_dir"].split("/")[-2], name=hps["log_dir"].split("/")[-1], config=hps, sync_tensorboard=True
        )

    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    Trainer = get_Trainer(hps)
    print("Seed: ", hps["seed"])
    trial = Trainer(hps)
    info_final = trial.run()

    if use_wandb:
        # wandb.log(prepend_keys(info_final,"final"))
        wandb.finish()

    print("\n\nFinal results:\n")
    print(info_final)

    print("\n\nTime elapsed: ", time.time() - start_time)
    return info_final


def get_Trainer(hps) -> StandardOnlineTrainer:
    if hps["task"]["name"] == "seh_frag":
        from gflownet.tasks.seh_frag import SEHFragTrainer
        return SEHFragTrainer
    elif hps["task"]["name"] in ["tdc_frag"]:
        from gflownet.tasks.tdc_frag import TDCFragTrainer
        return TDCFragTrainer
    elif hps["task"]["name"] in ["rna_bind"]:
        from gflownet.tasks.rna_bind import RNABindTrainer
        return RNABindTrainer
    else:
        raise ValueError(f"Unknown task!")


if __name__ == "__main__":
    hps = {
        "log_dir": f"./logs/test_no_buffer_lagging_0",
        "device": "cuda",
        "seed": 0,  # TODO: how is seed handled?
        "validate_every": 1000,  # 1000,
        "print_every": 10,
        "num_training_steps": 15650,
        "num_workers": 8,
        "num_final_gen_steps": 320,
        "overwrite_existing_exp": True,
        "exploration_helper": "no_exploration",
        "algo": {
            "method": "TB",
            "helper": "TB",
            "sampling_tau": 0.99,
            "sample_temp": 1.0,
            "online_batch_size": 64,
            "replay_batch_size": 32,
            "offline_batch_size": 0,
            "max_nodes": 9,
            "illegal_action_logreward": -75,
            "train_random_action_prob": 0.0,
            "valid_random_action_prob": 0.0,
            "train_random_traj_prob": 0.0,
            "valid_sample_cond_info": True,
            "tb": {
                "variant": TBVariant.TB,
                "Z_learning_rate": 1e-3,
                "Z_lr_decay": 50_000,
                "do_parameterize_p_b": False,
                "do_length_normalize": False,  ###TODO
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
            "capacity": 1000,
            "warmup": 100,
            "hindsight_ratio": 0.0,
            "insertion": {
                "strategy": "fifo",
                "sim_thresh": 0.7,
                "reward_thresh": 0.8,
            },
            "sampling": {
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
                "sample_dist": "constant",  # "uniform"
                "dist_params": [32.0],  # [0, 64.0],  #[16,32,64,96,128]
                "num_thermometer_dim": 1,
                "val_temp": 32.0,
            }
        },
        "task": {"name": "seh_frag", "helper": "seh_frag", "tdc": {"oracle": "qed"}},
        "evaluation": {
            "k": 100,
            "reward_thresh": 1,
            "distance_thresh": 0.3,
        },
    }

    conf = OmegaConf.create(hps)
    cli_conf = OmegaConf.from_cli()
    merged = OmegaConf.merge(conf, cli_conf)
    hps = OmegaConf.to_container(merged, resolve=True)
    if "use_wandb" in hps:
        use_wandb = hps["use_wandb"]
        del hps["use_wandb"]
    else:
        use_wandb = False
    if "exp_name" in hps:
        hps["log_dir"] = f"./logs/{hps['exp_name']}_{hps['seed']}"
        del hps["exp_name"]

    info_val = main(hps, use_wandb=use_wandb, entity="mokshjain")