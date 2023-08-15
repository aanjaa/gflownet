from typing import List

import numpy as np
import torch

from gflownet.config import Config


class ReplayBuffer(object):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):
        self.capacity = cfg.replay.capacity
        self.warmup = cfg.replay.warmup
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"

        self.buffer: List[tuple] = []
        self.position = 0
        self.rng = rng
        self.name = cfg.replay.name

    def push(self, *args): 
        #args: trajs, log_rewards, flat_rewards, cond_info, is_valid
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.name == "FIFO":
            idxs = self.rng.choice(len(self.buffer), batch_size)
        elif self.name == "beta_perc_from_top_alpha_perc_rewards":
            beta = 0.5 
            alpha = 0.1 
            batch_size_top = int(batch_size*beta)
            batch_size_all = batch_size - batch_size_top
            k = int(len(self.buffer)*alpha)
            # Sample some of the batch regularly
            idxs_all = self.rng.choice(len(self.buffer), int(batch_size/2))
            # Sample the rest from top 10% of the reward
            rewards = [x[2] for x in self.buffer]
            rewards = torch.tensor(rewards)
            values,indices = torch.topk(rewards,k)
            idxs_top = self.rng.choice(indices, batch_size-int(batch_size/2))
            # Concatenate the two
            idxs = np.concatenate((idxs_all,idxs_top))
        else:
            raise NotImplementedError

        out = list(zip(*[self.buffer[idx] for idx in idxs]))
        for i in range(len(out)):
            # stack if all elements are numpy arrays or torch tensors
            # (this is much more efficient to send arrays through multiprocessing queues)
            if all([isinstance(x, np.ndarray) for x in out[i]]):
                out[i] = np.stack(out[i], axis=0)
            elif all([isinstance(x, torch.Tensor) for x in out[i]]):
                out[i] = torch.stack(out[i], dim=0)
        return tuple(out)

    def __len__(self):
        return len(self.buffer)
