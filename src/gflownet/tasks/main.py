
import os
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union
import wandb

import numpy as np
import wandb
from gflownet.utils.metrics_final_eval import candidates_eval
from gflownet.utils.misc import prepend_keys
from gflownet.tasks.seh_frag import SEHFragTrainer
from gflownet.tasks.tdc_frag import TDCFragTrainer
from gflownet.config import Config
import torch


def main(hps,use_wandb=False):
    # hps must contain task.name, log_dir, overwrite_existing_exp

    if use_wandb:
        wandb.init(project=hps["log_dir"].split("/")[-2],name=hps["log_dir"].split("/")[-1],config=hps,sync_tensorboard=True)

    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    Trainer = get_Trainer(hps)
    trial = Trainer(hps)
    info_final = trial.run()
    if trial.cfg.num_final_gen_steps > 0:
        info_candidates = candidates_eval(path = hps["log_dir"]+"/final", k=100, thresh=0.7)
        info_final = {**info_final,**info_candidates}
    
    if use_wandb:
        wandb.log(prepend_keys(info_final,"final"))
        wandb.finish()
    
    #print(info_final)
    return info_final


def get_Trainer(hps) -> Union[SEHFragTrainer, TDCFragTrainer]:
    if hps["task"]["name"] == "seh_frag":
        return SEHFragTrainer
    elif hps["task"]["name"] == "tdc_frag":
        return TDCFragTrainer
    else:
        raise ValueError(f"Unknown task!")
    

if __name__ == "__main__":

    task_name = "tdc_frag" #"seh_frag"

    hps = {
        "log_dir": f"./logs/debug_{task_name}/",
        "device": "cuda"  if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 10, #10_000,
        "print_every": 10,
        "validate_every":10,
        "num_workers": 0,
        "num_final_gen_steps": 2,
        "top_k": 100,
        "opt": {
            "lr_decay": 20_000,
            },
        "algo": {
            "sampling_tau": 0.99,
            "global_batch_size": 64, #64,
            },
        "cond": {
            "temperature": {
                "sample_dist": "uniform",
                "dist_params": [0, 64.0],
                }
            },
        "task": {
            "name": task_name, 
            "tdc": {
                "oracle": "qed"
                },
            }
        }
    info_val = main(hps,use_wandb = True)

