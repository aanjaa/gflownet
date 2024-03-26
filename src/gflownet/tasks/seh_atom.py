import os
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol, ChiralType
from torch import Tensor
from torch.utils.data import Dataset

from gflownet.config import Config
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.tasks.seh_frag import SEHTask

class SEHAtomTrainer(StandardOnlineTrainer):
    task: SEHTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 5

        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10

        #cfg.algo.global_batch_size = 64
        #cfg.algo.offline_ratio = 0
        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 50
        cfg.algo.sampling_tau = 0.95
        cfg.algo.illegal_action_logreward = -256
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.train_random_traj_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        #cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = True

        cfg.model.num_emb = 128
        cfg.model.num_layers = 4
        cfg.model.graph_transformer.num_mlp_layers = 2

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

        cfg.cond.temperature.sample_dist = "constant"
        cfg.cond.temperature.dist_params = [96.0]

        cfg.task.name = "seh"

    def setup_task(self):
        self.task = SEHTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = MolBuildingEnvContext(
            ["C", "N", "O", "S", "F", "Cl", "Br"],
            charges=[0],
            chiral_types=[ChiralType.CHI_UNSPECIFIED],
            max_nodes=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            allow_5_valence_nitrogen=False,
            # We need to fix backward trajectories to use masks! should be fine now
            # And make sure the Nitrogen-related backward masks make sense
        )
        if hasattr(self.ctx, "graph_def"):
            self.env.graph_cls = self.ctx.graph_cls



def main():
    hps = {
        "log_dir": "./logs/debug_run_seh_atom",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 10_000,
        "num_workers": 8,
        "algo": {
            "replay_batch_size": 64,
            "online_batch_size": 64,
        },
        "replay" : {
            "use": True,
            "capacity": 10_000,
            "warmup": 100,
            "insertion": {
                "strategy": "diversity_and_reward",
            },
            "sampling": {
                "strategy": "uniform",
            },
        }
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHAtomTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()