import os
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from rdkit.Chem.QED import qed
from torch import Tensor
from torch.utils.data import Dataset

from gflownet.config import Config
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional


class SEHPlusTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self.cand_type = "mols"

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model, send_to_device=True)
        return {"seh": model}

    def sample_conditional_information(self, n: int, train_it: int, is_validation: bool = False) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n, is_validation)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu()
        preds[preds.isnan()] = 0
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        # import pdb; pdb.set_trace();
        qed_scores = np.array([qed(mol) for mol in mols])
        # import pdb; pdb.set_trace()
        preds = torch.as_tensor(qed_scores).float().reshape((-1, 1)) * preds
        return FlatRewards(preds), is_valid


class SEHPlusFragTrainer(StandardOnlineTrainer):
    task: SEHPlusTask

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
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.train_random_traj_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        #cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

        cfg.cond.temperature.sample_dist = "uniform"
        cfg.cond.temperature.dist_params = [0,64.0]

        cfg.task.name = "seh"

    def setup_task(self):
        self.task = SEHPlusTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes, num_cond_dim=self.task.num_cond_dim, min_len=self.cfg.algo.min_len
        )


# def main():
#     hps = {
#     "log_dir": "./logs/debug_run_seh_frag",
#     "experiment_name": "debug_run_seh_frag",
#     "device": "cuda"  if torch.cuda.is_available() else "cpu",
#     "overwrite_existing_exp": True,
#     "num_training_steps": 100, #10_000,
#     "validate_every":100,
#     "num_workers": 1,
#     "opt": {
#         "lr_decay": 20_000,
#         },
#     "algo": {
#         "sampling_tau": 0.99,
#         },
#     "cond": {
#         "temperature": {
#             "sample_dist": "uniform",
#             "dist_params": [0, 64.0],
#             }
#         },
#     "replay":{
#         "use": True
#         }
#     }

#     if os.path.exists(hps["log_dir"]):
#         if hps["overwrite_existing_exp"]:
#             shutil.rmtree(hps["log_dir"])
#         else:
#             raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
#     os.makedirs(hps["log_dir"])

#     trial = SEHFragTrainer(hps)
#     trial.print_every = 5
#     info_val = trial.run()


# if __name__ == "__main__":
#     main()


# #RAYTUNE VERSION
# def main(hps,use_wandb=False):
#     if use_wandb:
#         wandb.init(project=hps["log_dir"].split("/")[-2]+"_sweep",name=hps["log_dir"].split("/")[-1],config=hps,sync_tensorboard=True)

#     if os.path.exists(hps["log_dir"]):
#         if hps["overwrite_existing_exp"]:
#             shutil.rmtree(hps["log_dir"])
#         else:
#             raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
#     os.makedirs(hps["log_dir"])

#     trial = SEHFragTrainer(hps)
#     info_final = trial.run()
#     if trial.cfg.num_final_gen_steps > 0:
#         info_candidates = candidates_eval(path = hps["log_dir"]+"/final", k=100, thresh=0.7)
#         info_val = {**info_final,**info_candidates}
    
#     if use_wandb:
#         wandb.log(prepend_keys(info_final,"final"))
#         wandb.finish()
    
#     return info_val


# if __name__ == "__main__":

#     hps = {
#     "log_dir": "./logs/debug_run_seh_frag",
#     "device": "cuda"  if torch.cuda.is_available() else "cpu",
#     "overwrite_existing_exp": True,
#     "num_training_steps": 10, #10_000,
#     "print_every": 1,
#     "validate_every":10,
#     "num_workers": 0,
#     "num_final_gen_steps": 2 ,
#     "opt": {
#         "lr_decay": 20000,
#         },
#     "algo": {
#         "method": "TB",
#         "sampling_tau": 0.99,
#         "train_random_traj_prob": 0.2
#         },
#     "cond": {
#         "temperature": {
#             "sample_dist": "uniform",
#             "dist_params": [0, 64.0],
#             }
#         },
#     "task": {
#         "name": "seh"
#         }
#     }
#     info_val = main(hps,use_wandb= False)
def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": "./logs/debug_run_seh_frag",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 10_000,
        "num_workers": 8,
        "opt": {
            "lr_decay": 20000,
        },
        "algo": {"sampling_tau": 0.99},
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
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
