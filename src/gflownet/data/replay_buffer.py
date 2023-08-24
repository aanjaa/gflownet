from typing import List

import numpy as np
import torch

from gflownet.config import Config
from scipy.special import softmax
from gflownet.envs.graph_building_env import GraphBuildingEnvContext
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity as Similarity
from statistics import mean
import copy


class Trajectory:
    # args: (traj, log_reward, flat_reward, cond_info, is_valid)
    def __init__(self, args):
        self.traj = args[0]
        self.log_reward = args[1]
        self.flat_reward = args[2]
        self.cond_info = args[3]
        self.is_valid = args[4]
        self.smi = args[0]["smi"] if "smi" in args[0] else None
        self.fp = Chem.RDKFingerprint(Chem.MolFromSmiles(args[0]["smi"])) if "smi" in args[0] else None


class ReplayBuffer(object):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):  # , ctx: GraphBuildingEnvContext = None):
        self.capacity = cfg.replay.capacity
        self.warmup = cfg.replay.warmup  # Will not sample from buffer during warmup
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"
        # self.buffer: List[tuple] = []
        self.buffer: List[Trajectory] = []
        self.position = 0
        self.rng = rng
        self.sampling_strategy = cfg.replay.sampling_strategy
        # Diversity instertion strategies assume that we work with molecules
        self.insertion_strategy = cfg.replay.insertion_strategy
        self.reward_thresh = cfg.replay.reward_thresh
        self.sim_thresh = cfg.replay.sim_thresh

    def push(self, *args):
        traj = Trajectory(args)
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"

        # If buffer is not full, insert at the end
        if len(self.buffer) < self.capacity:
            # self.buffer.append(None)
            self.buffer.append(None)
            self.buffer[self.position] = traj
            self.position = (self.position + 1) % self.capacity

        # If buffer is full, insert according to insertion strategy
        else:
            if self.insertion_strategy == "fifo":
                self.buffer[self.position] = traj
                self.position = (self.position + 1) % self.capacity

            elif self.insertion_strategy == "reward_based":
                """
                Every inserted x increases the lowest reward of the buffer.
                """
                # flat rewards are not transformed with cond info
                if traj.flat_reward > self.min_reward:
                    self.buffer[self.min_reward_idx] = traj

            elif self.insertion_strategy == "diversity_based":
                """
                Insert if x is not too similar (fixed threshold) to any element in the buffer (replace spot randomly chosen).
                Still make a replacement in case the new candidate has a higher reward than the most similar molecule (replace spot: most similar molecule).
                """
                max_sim,max_sim_idx = self.compute_max_sim_with_buffer(traj)

                if max_sim < self.sim_thresh:
                        self.buffer[self.rng.choice(len(self.buffer))] = traj
                else:
                    # if the most similar molecule has a lower reward than the candidate molecule, replace it
                    if traj.flat_reward > self.buffer[max_sim_idx].flat_reward:
                        self.buffer[max_sim_idx] = traj

            elif self.insertion_strategy == "diversity_and_reward_based_fast":
                """
                Insert if x is not too similar (fixed threshold) to any element in the buffer and its reward is higher than lowest reward (replace spot: lowest reward)
                """
                max_sim,max_sim_idx = self.compute_max_sim_with_buffer(traj)

                if max_sim < self.sim_thresh and traj.flat_reward > self.min_reward:
                        self.buffer[self.min_reward_idx] = traj

            elif self.insertion_strategy == "diversity_and_reward_based":
                """
                Every inserted x satisfies two criteria: increase the lowest reward (unless the lowest reward is above a threshold) AND decrease the max similarity between any two elements in the buffer (unless all elements are already diverse enough).
                """
                max_sim,max_sim_idx = self.compute_max_sim_with_buffer(traj)

                if traj.flat_reward > min(self.min_reward,self.reward_thresh) and max_sim < max(self.max_sim_buffer,self.sim_thresh):
                    # Randomly choose to replace element with lowest reward or element with highest similarity
                    if self.rng.random() < 0.5:
                        self.buffer[self.min_reward_idx] = traj
                    else:
                        self.buffer[self.max_sim_buffer_idx] = traj                
            else:
                raise NotImplementedError
            
            print(self.get_buffer_stats())

    def compute_max_sim_with_buffer(self, traj):
        modes_fp = [traj.fp for traj in self.buffer]
        sim = Similarity(traj.fp, modes_fp)
        max_sim = max(sim)
        return max_sim, max_sim.index(max_sim)

    @property
    def min_reward(self):
        flat_rewards = torch.stack([traj.flat_reward for traj in self.buffer])
        return torch.min(flat_rewards)
    
    @property
    def avg_reward(self):
        flat_rewards = torch.stack([traj.flat_reward for traj in self.buffer])
        return torch.mean(flat_rewards)
    
    @property
    def min_reward_idx(self):
        flat_rewards = torch.stack([traj.flat_reward for traj in self.buffer])
        return torch.argmin(flat_rewards)

    @property
    def max_sim_buffer(self):
        """
        Finds the molecule with the highest tanimoto similarity to any other molecule in the buffer.
        """
        avg_sim, max_sim, min_sim = self.compute_sim_within_buffer()
        return max(max_sim)

    @property
    def max_sim_buffer_idx(self):
        """
        Index of molecule with the highest tanimoto similarity to any other molecule in the buffer.
        """
        avg_sim, max_sim, min_sim = self.compute_sim_within_buffer()
        return max_sim.index(self.max_sim)


    def compute_sim_within_buffer(self):
        """
        Returns a list of the average, max and min tanimoto similarities between each molecule and all the others in the buffer.
        """
        modes_fp = [traj.fp for traj in self.buffer]
        avg_sim = []
        max_sim = []
        min_sim = []

        for i, traj in enumerate(self.buffer):
            # remove itself from the list of modes
            modes_fp_wo_self = copy.deepcopy(modes_fp)
            modes_fp_wo_self = modes_fp_wo_self[:i] + modes_fp_wo_self[i + 1 :]
            sim = Similarity(traj.fp, modes_fp_wo_self)
            avg_sim.append(sum(sim) / len(sim))
            max_sim.append(max(sim))
            min_sim.append(min(sim))

        return avg_sim, max_sim, min_sim

    
    def get_buffer_stats(self):
        avg_sim, max_sim, min_sim = self.compute_inter_buffer_sim()
        # return dictionary of metrics, rounded to three decimal points
        return {
            "lowest_reward": round(float(self.min_reward), 3),
            "avg_reward": round(float(self.avg_reward()), 3),
            "max_max_sim": round(self.max_sim, 3),
            "avg_max_sim": round(mean(max_sim), 3),
            "avg_avg_sim": round(mean(avg_sim), 3)
        }

    def sample(self, batch_size):
        if self.sampling_strategy == "uniform":
            idxs = self.rng.choice(len(self.buffer), batch_size)

        elif self.sampling_strategy == "weighted":
            rewards = torch.tensor([x[2] for x in self.buffer])
            p = softmax(rewards, axis=0)
            idxs = self.rng.choice(len(self.buffer), batch_size, p=p)

        elif self.sampling_strategy == "top_quantile_sampling":
            beta = 0.5
            alpha = 0.1
            batch_size_top = int(batch_size * beta)
            batch_size_all = batch_size - batch_size_top
            k = int(len(self.buffer) * alpha)
            # Sample some of the batch regularly
            idxs_all = self.rng.choice(len(self.buffer), int(batch_size / 2))
            # Sample the rest from top 10% of the reward
            flat_rewards = torch.tensor([traj.flat_reward for traj in self.buffer])
            _, indices = torch.topk(flat_rewards, k)
            idxs_top = self.rng.choice(indices, batch_size - int(batch_size / 2))
            # Concatenate the two
            idxs = np.concatenate((idxs_all, idxs_top))
        else:
            raise NotImplementedError

        replay_trajs = tuple([self.buffer[idx].traj for idx in idxs])
        replay_logr = torch.stack([self.buffer[idx].log_reward for idx in idxs])
        replay_fr = torch.stack([self.buffer[idx].flat_reward for idx in idxs])
        replay_condinfo = tuple([self.buffer[idx].cond_info for idx in idxs])
        replay_valid = torch.stack([self.buffer[idx].is_valid for idx in idxs])

        return replay_trajs, replay_logr, replay_fr, replay_condinfo, replay_valid

        # out = list(zip(*[self.buffer[idx] for idx in idxs]))
        # for i in range(len(out)):
        #     # stack if all elements are numpy arrays or torch tensors
        #     # (this is much more efficient to send arrays through multiprocessing queues)
        #     if all([isinstance(x, np.ndarray) for x in out[i]]):
        #         out[i] = np.stack(out[i], axis=0)
        #     elif all([isinstance(x, torch.Tensor) for x in out[i]]):
        #         out[i] = torch.stack(out[i], dim=0)

        # return tuple(out) #replay_trajs, replay_logr, replay_fr, replay_condinfo, replay_valid

    def __len__(self):
        return len(self.buffer)
