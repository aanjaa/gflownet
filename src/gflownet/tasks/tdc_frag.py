import ast
import copy
from typing import Any, Callable, Dict, List, Tuple, Union
import os
import shutil

import numpy as np
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from rdkit.Chem.rdmolfiles import MolToSmiles
import scipy.stats as stats
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
import torch_geometric.data as gd

from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.models import bengio2021flow
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.config import Config
from gflownet.utils.transforms import thermometer
from gflownet.utils.conditioning import TemperatureConditional


import socket
from tdc import Oracle

class TDCTask(GFNTask):
    """
    Task where the reward is computed using oracles from TDC
    """
    # def __init__(self, name: str, temperature_distribution: str, temperature_parameters: Tuple[float]):
    #     self.oracle = Oracle(name=name)
    #     self.temperature_sample_dist = temperature_distribution
    #     self.temperature_dist_params = temperature_parameters

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self.oracle = Oracle(cfg.task.tdc.oracle)
        self.cand_type = "mols"

    
    # def sample_conditional_information(self, n):
    #     beta = None
    #     if self.temperature_sample_dist == 'gamma':
    #         loc, scale = self.temperature_dist_params
    #         beta = self.rng.gamma(loc, scale, n).astype(np.float32)
    #         upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
    #     elif self.temperature_sample_dist == 'uniform':
    #         beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
    #         upper_bound = self.temperature_dist_params[1]
    #     elif self.temperature_sample_dist == 'beta':
    #         beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
    #         upper_bound = 1
    #     beta_enc = thermometer(torch.tensor(beta), 32, 0, upper_bound)  # TODO: hyperparameters
    #     return {'beta': torch.tensor(beta), 'encoding': beta_enc}
    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    # def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
    #     if isinstance(flat_reward, list):
    #         flat_reward = torch.tensor(flat_reward)
    #     return flat_reward.flatten()**cond_info['beta']
    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        smiles = [MolToSmiles(i) for i in mols]
        is_valid = torch.tensor([i is not None or i != '' for i in smiles]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        scores = torch.tensor(self.oracle(smiles)).reshape((-1, 1))    
        #print(scores)
        scores = scores.to(dtype=torch.float32)
        return FlatRewards(scores), is_valid


class TDCFragTrainer(StandardOnlineTrainer):
    task: TDCTask

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

        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"

        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.0 #0.9

        cfg.algo.illegal_action_logreward = -75

        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0

        #fg.algo.valid_offline_ratio = 0

        cfg.algo.tb.epsilon = None

        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 20_000

        cfg.algo.tb.do_parameterize_p_b = False
        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

        cfg.cond.temperature.sample_dist = "uniform"
        cfg.cond.temperature.dist_params = [.5,32.0]

        cfg.task.name = "qed"
        #cfg.oracle_name = 'celecoxib_rediscovery'

        #https://github.com/mims-harvard/TDC/blob/main/tutorials/TDC_105_Oracle.ipynb
        # "QED", "Personalized LogP", "DRD2", "GSK3", "JNK3", "SA", "aripiprazole_similarit"

        #('jnk3' 'gsk3b' 'celecoxib_rediscovery' \\
        # 'troglitazone_rediscovery' \\
        # 'thiothixene_rediscovery' 'albuterol_similarity' 'mestranol_similarity' \\
        # 'isomers_c7h8n2o2' 'isomers_c9h10n2o2pf2cl' 'median1' 'median2' 'osimertinib_mpo' \\
        # 'fexofenadine_mpo' 'ranolazine_mpo' 'perindopril_mpo' 'amlodipine_mpo' \\
        # 'sitagliptin_mpo' 'zaleplon_mpo' 'valsartan_smarts' 'deco_hop' 'scaffold_hop' 'qed' 'drd2')
    
    def setup_task(self):
        self.task = TDCTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )
    # def setup_task(self):
    #     self.task = TDCTask(self.hps['oracle_name'], self.hps['temperature_sample_dist'],
    #                         ast.literal_eval(self.hps['temperature_dist_params']))

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(max_frags=self.cfg.algo.max_nodes, num_cond_dim=self.task.num_cond_dim)
 

    # def setup_model(self):
    #     self.model = GraphTransformerGFN(self.ctx, num_emb=self.hps['num_emb'], num_layers=self.hps['num_layers'])

    # def setup(self):
    #     hps = self.hps
    #     RDLogger.DisableLog('rdApp.*')
    #     self.rng = np.random.default_rng(142857)
    #     self.env = GraphBuildingEnv()
    #     self.ctx = FragMolBuildingEnvContext(max_frags=9, num_cond_dim=hps['num_cond_dim'])
    #     self.training_data = []
    #     self.test_data = []
    #     self.offline_ratio = 0
    #     self.valid_offline_ratio = 0
    #     self.setup_algo()
    #     self.setup_task()
    #     self.setup_model()

    #     # Separate Z parameters from non-Z to allow for LR decay on the former
    #     Z_params = list(self.model.logZ.parameters())
    #     non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
    #     self.opt = torch.optim.Adam(non_Z_params, hps['learning_rate'], (hps['momentum'], 0.999),
    #                                 weight_decay=hps['weight_decay'], eps=hps['adam_eps'])
    #     self.opt_Z = torch.optim.Adam(Z_params, hps['learning_rate'], (0.9, 0.999))
    #     self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2**(-steps / hps['lr_decay']))
    #     self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(self.opt_Z, lambda steps: 2**(-steps / hps['Z_lr_decay']))

    #     self.sampling_tau = hps['sampling_tau']
    #     if self.sampling_tau > 0:
    #         self.sampling_model = copy.deepcopy(self.model)
    #     else:
    #         self.sampling_model = self.model
    #     eps = hps['tb_epsilon']
    #     hps['tb_epsilon'] = ast.literal_eval(eps) if isinstance(eps, str) else eps

    #     self.mb_size = hps['global_batch_size']
    #     self.clip_grad_param = hps['clip_grad_param']
    #     self.clip_grad_callback = {
    #         'value': (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
    #         'norm': (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
    #         'none': (lambda x: None)
    #     }[hps['clip_grad_type']]

    # def step(self, loss: Tensor):
    #     loss.backward()
    #     for i in self.model.parameters():
    #         self.clip_grad_callback(i)
    #     self.opt.step()
    #     self.opt.zero_grad()
    #     self.opt_Z.step()
    #     self.opt_Z.zero_grad()
    #     self.lr_sched.step()
    #     self.lr_sched_Z.step()
    #     if self.sampling_tau > 0:
    #         for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
    #             b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))

## STANDARD VERSION
# def main():
#     """Example of how this model can be run outside of Determined"""
#     hps = {
#         "log_dir": "./logs/debug_run_seh_frag",
#         "experiment_name": "debug_run_seh_frag",
#         "device": "cuda"  if torch.cuda.is_available() else "cpu",
#         "overwrite_existing_exp": True,
#         "num_training_steps": 10, #10_000,
#         "validate_every": 10,
#         "num_workers": 1,
#         "opt": {
#             "lr_decay": 20_000,
#             },
#         "algo": {
#             "sampling_tau": 0.99,
#             },
#         "cond": {
#             "temperature": {
#                 "sample_dist": "uniform",
#                 "dist_params": [0, 64.0],
#                 }
#             },
#         "task": {
#             "tdc": {
#                 "oracle_name": "mestranol_similarity",
#                 }
#             },
#         }
    
#     trial = TDCFragTrainer(hps)
#     trial.print_every = 5
#     info_val = trial.run()


# if __name__ == '__main__':
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

#     trial = TDCFragTrainer(hps)
#     info_val = trial.run()
#     if use_wandb:
#         wandb.log(prepend_keys(info_val,"final"))
#         wandb.finish()
#     return info_val



# if __name__ == "__main__":

#     hps = {
#         "log_dir": "./logs/debug_tdc_opt_frag",
#         "device": "cuda"  if torch.cuda.is_available() else "cpu",
#         "overwrite_existing_exp": True,
#         "num_training_steps": 10, #10_000,
#         "print_every": 10,
#         "validate_every":10,
#         "num_workers": 1,
#         "num_final_gen_steps": 2 ,
#         "opt": {
#             "lr_decay": 20_000,
#             },
#         "algo": {
#             "sampling_tau": 0.99,
#             },
#         "cond": {
#             "temperature": {
#                 "sample_dist": "uniform",
#                 "dist_params": [0, 64.0],
#                 }
#             },
#         "task": {
#             "name": "qed"
#             }
#         }
#     info_val = main(hps,use_wandb = True)