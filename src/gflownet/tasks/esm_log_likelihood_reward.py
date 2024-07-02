from esm_reward.lm_design import Designer
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union
from copy import deepcopy
from torch.utils.data import TensorDataset
from torchtyping import TensorType
import torch.nn.functional as F
import pandas as pd
import numpy as np
import shutil
import socket
import torch
import os

from gflownet.config import Config
from gflownet.envs.seq_building_env import AutoregressiveSeqBuildingContext, SeqBuildingEnv
from gflownet.models.seq_transformer import SeqTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional

AMINO_ACID_VOCAB = list('ILVAGMFYWEDQNHCRKSTP')
AMINO_ACID_VOCAB = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W']
_vocab = deepcopy(AMINO_ACID_VOCAB)
#_vocab.remove('C')

_REPO_IDX_TO_CHAR = {idx: char for idx, char in enumerate(_vocab)}
_SUPPRESS_AAS = {'C'}

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _expand_states(states, padding_token, eos_token):
    actions, seqs = [], []
    arange_idx = torch.arange(len(states))
    for idx in range(states.shape[1] - 1, -1, -1):
        actions.append(states[:, idx].clone())

        if idx < states.shape[1] - 1:
            states[:, idx + 1] = padding_token

        states[:, idx] = eos_token
        seqs.append(states.clone())

    seqs = torch.stack(tuple(reversed(seqs)), dim=1)
    actions = torch.stack(tuple(reversed(actions)), dim=1)

    return seqs, actions

class ESMRewardModelWrapper(Designer):
    def __init__(
        self,
        seq_len: int,
        language_model_energy_term_weight: float,
        ngram_energy_term_weight: float,
        ngram_orders: List[int]
    ):
        torch.nn.Module.__init__(self)

        self.allowed_AA = ''.join(
            AA
            for AA in self.standard_AA
            if not AA in _SUPPRESS_AAS
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_models()
        self._init_no_target(seq_len)

        self.language_model_energy_term_weight = language_model_energy_term_weight
        self.ngram_energy_term_weight = ngram_energy_term_weight
        self.ngram_orders = ngram_orders

        self.all_esm_toks = self.vocab.all_toks
        self.esm_vocab_char_to_idx = {
            char: idx
            for idx, char in enumerate(self.all_esm_toks)
            if char in self.allowed_AA
        }

    def _encode(
        self,
        sequences: 'TensorType["batch_size", "seq_len", int]'
    ) -> 'TensorType["batch_size", "seq_len", "num_aa_types", int]':
        def convert(token):
            return self.esm_vocab_char_to_idx[token]

        # A token could be invalid, specifically if the token is an
        # end-of-sentence or padding token
        def is_valid(token):
            return token in self.esm_vocab_char_to_idx

        big_list = [
            [convert(tkn) for tkn in seq if is_valid(tkn)]
            for seq in sequences
        ]

        int_esm_encoded_seqs = torch.tensor(
            [
                [convert(tkn) for tkn in seq if is_valid(tkn)]
                for seq in sequences
            ],
            device=get_device()
        )

        return F.one_hot(int_esm_encoded_seqs, len(self.all_esm_toks)).float()

    def calc_total_loss(self, sequences: List[str]):
    #    LM_w,
    #    struct_w,
    #    ngram_w,
    #    ngram_orders,
    #    temp_struct=None
    #):
        return super().calc_total_loss(
            x=self._encode(sequences),
            mask=None,
            LM_w=self.language_model_energy_term_weight,
            struct_w=False,
            ngram_w=self.ngram_energy_term_weight,
            ngram_orders=self.ngram_orders
        )[0]

class ESMLogLikelihoodTask(GFNTask):
    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator,
        wrap_model: Callable[[torch.nn.Module], torch.nn.Module] = None,
    ):
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self._wrap_model = wrap_model

        esm_reward_calculator = ESMRewardModelWrapper(
            cfg.algo.max_len,
            cfg.task.esm_log_likelihood.language_model_energy_term_weight,
            cfg.task.esm_log_likelihood.ngram_energy_term_weight,
            cfg.task.esm_log_likelihood.ngram_orders
        )
        self.esm_reward_calculator, self.device = self._wrap_model(
            esm_reward_calculator,
            send_to_device=True
        )

        self.cand_type = "seqs"

    def sample_conditional_information(
        self,
        n: int,
        train_it: int,
        is_validation: bool = False
    ) -> Dict[str, torch.Tensor]:
        return self.temperature_conditional.sample(n, is_validation)

    def cond_info_to_logreward(self, cond_info: Dict[str, torch.Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, objs: List[str]) -> Tuple[FlatRewards, torch.Tensor]:
        log_rewards = self.get_log_rewards(objs)[0]

        return FlatRewards(log_rewards[:, None]), torch.ones(len(objs), dtype=torch.bool)

    def get_log_rewards(
        self,
        sequences: List[str]
    ) -> TensorType["batch_size", float]:
        return self.esm_reward_calculator.calc_total_loss(sequences=sequences),
        #    mask=None,
        #    LM_w=self.language_model_energy_term_weight,
        #    struct_w=False,
        #    ngram_w=self.ngram_energy_term_weight,
        #    ngram_orders=self.ngram_orders
        #)

class ESMLogLikelihoodTrainer(StandardOnlineTrainer):
    task: ESMLogLikelihoodTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 0
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.model.num_emb = 64
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 30
        cfg.algo.max_len = 30
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-2
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

    def setup_model(self):
        self.model = SeqTransformerGFN(
            self.ctx,
            self.cfg,
            min_len=30
        )

    def setup_task(self):
        self.task = ESMLogLikelihoodTask(
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.env = SeqBuildingEnv(None)
        self.ctx = AutoregressiveSeqBuildingContext(
            AMINO_ACID_VOCAB,
            self.task.num_cond_dim,
        )

    def setup_algo(self):
        super().setup_algo()
        # If the algo implements it, avoid giving, ["A", "AB", "ABC", ...] as a sequence of inputs, and instead give
        # "ABC...Z" as a single input, but grab the logits at every timestep. Only works if using a transformer with
        # causal self-attention.
        self.algo.model_is_autoregressive = True


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": "./logs/debug_run_esm",
        "device": "cuda",
        "overwrite_existing_exp": True,
        "num_training_steps": 2_000,
        "checkpoint_every": 200,
        "num_workers": 4,
        "cond": {
            "temperature": {
                "sample_dist": "constant",
                "dist_params": [2.0],
                "num_thermometer_dim": 1,
            }
        },
        "algo": {"train_random_action_prob": 0.05},
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = ESMLogLikelihoodTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
