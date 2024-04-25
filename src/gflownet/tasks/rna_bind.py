import os
import shutil
import socket
from typing import Dict, List, Tuple
# import RNA
import flexs
import numpy as np
import torch
from torch import Tensor


from gflownet.config import Config
from gflownet.envs.seq_building_env import AutoregressiveSeqBuildingContext, SeqBuildingEnv
from gflownet.models.seq_transformer import SeqTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional

# class RNABinding(flexs.Landscape):
#     """RNA binding landscape using ViennaRNA `duplexfold`."""

#     def __init__(
#         self,
#         targets: List[str],
#         seq_length: int,
#         conserved_region: Dict = None,
#     ):
#         """
#         Create an RNABinding landscape.

#         Args:
#             targets: List of RNA strings that will be binding targets.
#                 If more than one, the fitness score is the mean of each binding fitness.
#             seq_length: Length of sequences to be evaluated.
#             conserved_region: A dictionary of the form `{start: int, pattern: str}`
#                 defining the start of the conserved region and the pattern that must be
#                 conserved. Sequences violating these criteria will receive a score of
#                 zero (useful for creating `swampland` areas).

#         """
#         # ViennaRNA is not available through pip, so give a warning message
#         # if not installed.
#         try:
#             RNA
#         except NameError as e:
#             raise ImportError(
#                 f"{e}.\n"
#                 "Hint: ViennaRNA not installed.\n"
#                 "      Source and binary installations available at "
#                 "https://www.tbi.univie.ac.at/RNA/#download.\n"
#                 "      Conda installation available at "
#                 "https://anaconda.org/bioconda/viennarna."
#             ) from e

#         super().__init__(name=f"RNABinding_T{targets}_L{seq_length}")

#         self.targets = targets
#         self.seq_length = seq_length
#         self.conserved_region = conserved_region
#         self.norm_values = self.compute_min_binding_energies()

#         self.sequences = {}

#     def compute_min_binding_energies(self):
#         """Compute the lowest possible binding energy for each target."""
#         complements = {"A": "U", "C": "G", "G": "C", "U": "A"}

#         min_energies = []
#         for target in self.targets:
#             complement = "".join(complements[x] for x in target)[::-1]
#             energy = RNA.duplexfold(complement, target).energy
#             min_energies.append(energy * self.seq_length / len(target))

#         return np.array(min_energies)

#     def _fitness_function(self, sequences):
#         fitnesses = []

#         for seq in sequences:

#             # Check that sequence is of correct length
#             if len(seq) != self.seq_length:
#                 raise ValueError(
#                     f"All sequences in `sequences` must be of length {self.seq_length}"
#                 )

#             # If `self.conserved_region` is not None, check that the region is conserved
#             if self.conserved_region is not None:
#                 start = self.conserved_region["start"]
#                 pattern = self.conserved_region["pattern"]

#                 # If region not conserved, fitness is 0
#                 if seq[start : start + len(pattern)] != pattern:
#                     fitnesses.append(0)
#                     continue

#             # Energy is sum of binding energies across all targets
#             energies = np.array(
#                 [RNA.duplexfold(target, seq).energy for target in self.targets]
#             )
#             fitness = (energies / self.norm_values).mean()

#             fitnesses.append(fitness)

#         return np.array(fitnesses)


CHAR_MAP = ['A', 'C', 'G', 'T']

class RNABindingTask(GFNTask):
    """Sets up a task where the reward is the number of times some sequences appear in the input. Normalized to be
    in [0,1]"""

    def __init__(
        self,
        task: str,
        cfg: Config,
        rng: np.random.Generator,
    ):
        self.task_name = task
        problem = flexs.landscapes.rna.registry()[self.task_name]
        self.task = flexs.landscapes.RNABinding(**problem['params'])

        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        self.cand_type = "seqs"
        # self.norm = cfg.algo.max_len / min(map(len, seqs))

    def sample_conditional_information(self, n: int, train_it: int, is_validation: bool = False) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n, is_validation)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, objs: List[str]) -> Tuple[FlatRewards, Tensor]:
        # print(objs)
        rs = torch.tensor(self.task.get_fitness(objs)).float()
        return FlatRewards(rs[:, None]), torch.ones(len(objs), dtype=torch.bool)


class RNABindTrainer(StandardOnlineTrainer):
    task: RNABindingTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        # cfg.algo.offline_ratio = 0
        cfg.model.num_emb = 64
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 14
        cfg.algo.min_len = 14
        cfg.algo.max_len = 14
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        # cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-2
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

    def setup_model(self):
        self.model = SeqTransformerGFN(
            self.ctx,
            self.cfg,
            min_len=self.cfg.algo.min_len
        )

    def setup_task(self):
        self.task = RNABindingTask(
            "L14_RNA1",
            cfg=self.cfg,
            rng=self.rng,
        )

    def setup_env_context(self):
        self.env = SeqBuildingEnv(None)
        self.ctx = AutoregressiveSeqBuildingContext(
            "ACTG",
            self.task.num_cond_dim,
            self.cfg.algo.min_len,
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
        "log_dir": "./logs/debug_run_toy_seq",
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
        "algo": {
            "train_random_action_prob": 0.05,

        },
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = ToySeqTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
