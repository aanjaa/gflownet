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
import heapq


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

    # To make the object comparable for heapq
    def __lt__(self, other):
        return self.flat_reward < other.flat_reward


class ReplayBuffer(object):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):  # , ctx: GraphBuildingEnvContext = None):
        self.capacity = cfg.replay.capacity
        self.warmup = cfg.replay.warmup  # Will not sample from buffer during warmup
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"
        # self.buffer: List[tuple] = []
        self.buffer: List[(float,Trajectory)] = []
        self.position = 0
        self.rng = rng
        # Diversity instertion strategies assume that we work with molecules
        self.insertion_strategy = cfg.replay.insertion.strategy
        self.sim_thresh = cfg.replay.insertion.sim_thresh
        self.reward_thresh = cfg.replay.insertion.reward_thresh
        self.sampling_strategy = cfg.replay.sampling.strategy
        self.alpha = cfg.replay.sampling.quantile.alpha
        self.beta = cfg.replay.sampling.quantile.beta
        self.reward_power = cfg.replay.sampling.weighted.reward_power


    def push(self, *args):
        traj = Trajectory(args)
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"
        
        # First fill buffer
        if len(self.buffer) < self.capacity:
            if self.insertion_strategy in ["fifo","diversity"]:
                heapq.heappush(self.buffer, (self.position,traj))
                self.position += 1

            if self.insertion_strategy in  ["reward","diversity_and_reward"]:
                heapq.heappush(self.buffer, (traj.flat_reward,traj))

        # If buffer is full, insert according to insertion strategy
        else:
            if self.insertion_strategy == "fifo":
                heapq.heapreplace(self.buffer, (self.position, traj))
                self.position += 1

            elif self.insertion_strategy == "reward":
                """
                Every inserted x increases the lowest reward of the buffer.
                Replace spot: lowest reward element
                """
                if traj.flat_reward > self.buffer[0][0]:
                    heapq.heapreplace(self.buffer, (traj.flat_reward,traj))

            elif self.insertion_strategy == "diversity":
                """
                Insert if x is not too similar (fixed threshold) to any element in the buffer (replace spot: randomly chosen).
                """
                max_sim = self.compute_max_sim_with_buffer(traj)

                if max_sim < self.sim_thresh:
                    rand_int = self.rng.integers(self.capacity)
                    self.buffer[rand_int] = (rand_int,traj)
    

            elif self.insertion_strategy == "diversity_and_reward":
                """
                Insert if x is not too similar (fixed threshold) to any element in the buffer and its reward is higher than lowest reward
                Replace spot: lowest reward element
                """
                max_sim = self.compute_max_sim_with_buffer(traj)

                if max_sim < self.sim_thresh and traj.flat_reward > self.buffer[0][0]:
                    heapq.heapreplace(self.buffer, (traj.flat_reward,traj))
             
            else:
                raise NotImplementedError
            

    def compute_max_sim_with_buffer(self, traj):
        """
        Compute similarity between traj and all elements in the buffer.
        """
        modes_fp = [traj.fp for (_,traj) in self.buffer]
        if all([v is None for v in modes_fp]):
            return 0
        sim = Similarity(traj.fp, modes_fp)
        max_sim = max(sim)
        return max_sim #, sim.index(max_sim)

    @property
    def min_reward(self):
        flat_rewards = torch.stack([traj.flat_reward for (_,traj) in self.buffer])
        return torch.min(flat_rewards)
    
    @property
    def avg_reward(self):
        flat_rewards = torch.stack([traj.flat_reward for (_,traj) in self.buffer])
        return torch.mean(flat_rewards)

    def compute_simlarities_within_buffer(self):
        """
        Returns a list of the average, max and min tanimoto similarities between each molecule and all the others in the buffer.
        """
        modes_fp = [traj.fp for (_,traj) in self.buffer]
        avg_sim_list = []
        max_sim_list = []
        min_sim_list = []

        for i, (_,traj) in enumerate(self.buffer):
            # remove itself from the list of modes
            modes_fp_wo_self = copy.deepcopy(modes_fp)
            modes_fp_wo_self = modes_fp_wo_self[:i] + modes_fp_wo_self[i + 1 :]
            sim = Similarity(traj.fp, modes_fp_wo_self)
            avg_sim_list.append(sum(sim) / len(sim))
            max_sim_list.append(max(sim))
            min_sim_list.append(min(sim))

        return avg_sim_list, max_sim_list, min_sim_list
    
    
    def get_buffer_stats(self):
        avg_sim_list, max_sim_list, min_sim_list = self.compute_simlarities_within_buffer()
        # return dictionary of metrics, rounded to three decimal points
        return {
            "lowest_reward": round(float(self.min_reward), 3),
            "avg_reward": round(float(self.avg_reward), 3),
            "max of max_sim": round(max(max_sim_list), 3),
            "avg of max_sim": round(mean(max_sim_list), 3),
            "avg of avg_sim": round(mean(avg_sim_list), 3)
        }

    def sample(self, batch_size):

        #print(self.get_buffer_stats())
        # import pdb; pdb.set_trace();
        if len(self.buffer) < batch_size:
            return [], torch.tensor([]), torch.tensor([]), [], torch.tensor([])
        if self.sampling_strategy == "uniform":
            idxs = self.rng.choice(len(self.buffer), batch_size)

        elif self.sampling_strategy == "weighted":
            rewards = torch.tensor([traj.flat_reward for traj in self.buffer])
            # Raise reward to the power of reward_power
            rewards = rewards ** self.reward_power
            # Get probabilities proportional to rewards
            p = softmax(rewards, axis=0)
            idxs = self.rng.choice(len(self.buffer), batch_size, p=p)

        elif self.sampling_strategy == "quantile":
            """
            make β% of batch come from top α% of data (e.g. half the batch come from top 10% of data).
            """
            batch_size_top = int(batch_size * self.beta)
            batch_size_all = batch_size - batch_size_top
            k = int(len(self.buffer) * self.alpha)
            # Sample some of the batch regularly
            idxs_all = self.rng.choice(len(self.buffer), batch_size_all)
            # Sample the rest from top 10% of the reward
            flat_rewards = torch.tensor([traj.flat_reward for (_,traj) in self.buffer])
            _, indices = torch.topk(flat_rewards, k)
            idxs_top = self.rng.choice(indices, batch_size_top)
            # Concatenate the two
            idxs = np.concatenate((idxs_all, idxs_top))
        else:
            raise NotImplementedError

        replay_trajs = tuple([self.buffer[idx][1].traj for idx in idxs])
        replay_logr = torch.stack([self.buffer[idx][1].log_reward for idx in idxs])
        replay_fr = torch.stack([self.buffer[idx][1].flat_reward for idx in idxs])
        replay_condinfo = tuple([self.buffer[idx][1].cond_info for idx in idxs])
        replay_valid = torch.stack([self.buffer[idx][1].is_valid for idx in idxs])

        return replay_trajs, replay_logr, replay_fr, replay_condinfo, replay_valid

    def __len__(self):
        return len(self.buffer)
