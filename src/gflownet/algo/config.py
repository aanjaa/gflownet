from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TBVariant(Enum):
    """See algo.trajectory_balance.TrajectoryBalance for details."""

    TB = 0
    SubTB1 = 1
    DB = 2


@dataclass
class TBConfig:
    """Trajectory Balance config.

    Attributes
    ----------
    bootstrap_own_reward : bool
        Whether to bootstrap the reward with the own reward. (deprecated)
    epsilon : Optional[float]
        The epsilon parameter in log-flow smoothing (see paper)
    reward_loss_multiplier : float
        The multiplier for the reward loss when bootstrapping the reward. (deprecated)
    variant : TBVariant
        The loss variant. See algo.trajectory_balance.TrajectoryBalance for details.
    do_correct_idempotent : bool
        Whether to correct for idempotent actions
    do_parameterize_p_b : bool
        Whether to parameterize the P_B distribution (otherwise it is uniform)
    do_length_normalize : bool
        Whether to normalize the loss by the length of the trajectory
    subtb_max_len : int
        The maximum length trajectories, used to cache subTB computation indices
    Z_learning_rate : float
        The learning rate for the logZ parameter (only relevant when do_subtb is False)
    Z_lr_decay : float
        The learning rate decay for the logZ parameter (only relevant when do_subtb is False)
    """

    bootstrap_own_reward: bool = False
    epsilon: Optional[float] = None
    reward_loss_multiplier: float = 1.0
    variant: TBVariant = TBVariant.TB
    do_correct_idempotent: bool = False
    do_parameterize_p_b: bool = False
    do_length_normalize: bool = False
    subtb_max_len: int = 128
    Z_learning_rate: float = 1e-4
    Z_lr_decay: float = 50_000
    cum_subtb: bool = True


@dataclass
class MOQLConfig:
    gamma: float = 1
    num_omega_samples: int = 32
    num_objectives: int = 2
    lambda_decay: int = 10_000
    penalty: float = -10


@dataclass
class A2CConfig:
    entropy: float = 0.01
    gamma: float = 1
    penalty: float = -10


@dataclass
class FMConfig:
    epsilon: float = 1e-38
    balanced_loss: bool = False
    leaf_coef: float = 10
    correct_idempotent: bool = False


@dataclass
class SQLConfig:
    alpha: float = 0.01
    gamma: float = 1
    penalty: float = -10


@dataclass
class AlgoConfig:
    """Generic configuration for algorithms

    Attributes
    ----------
    method : str
        The name of the algorithm to use (e.g. "TB")
    online_batch_size : int
        The batch size sampled from the model.
    replay_batch_size : int
        The batch size sampled from the replay buffer.
    offline_batch_size : int
        The batch size sampled from the offline dataset.
    global_batch_size : int
        The batch size for training
    min_len: int
        If >0, prevents the agent from using the Stop action before min_len steps (trajectories may still end for
        other reasons, but generally setting min_len==max_len should produce fixed length trajectories).
    max_len : int
        The maximum length of a trajectory
    max_nodes : int
        The maximum number of nodes in a generated graph
    max_edges : int
        The maximum number of edges in a generated graph
    illegal_action_logreward : float
        The log reward an agent gets for illegal actions
    train_random_action_prob : float
        The probability of taking a random action during training
    valid_random_action_prob : float
        The probability of taking a random action during validation
    valid_sample_cond_info : bool
        Whether to sample conditioning information during validation (if False, expects a validation set of cond_info)
    sampling_tau : float
        The EMA factor for the sampling model (theta_sampler = tau * theta_sampler + (1-tau) * theta)
    """

    method: str = "TB"
    helper: str = "TB"
    online_batch_size: int = 64
    replay_batch_size: int = 0
    offline_batch_size: int = 0
    global_batch_size: int = 64
    min_len: int = 0
    max_len: int = 128
    max_nodes: int = 128
    max_edges: int = 128
    illegal_action_logreward: float = -100
    train_random_action_prob: float = 0.0
    train_random_traj_prob: float = 0.0
    valid_random_action_prob: float = 0.0
    valid_sample_cond_info: bool = True
    sampling_tau: float = 0.0
    sample_temp: float = 1.0
    reset_schedule: Optional[list] = None
    reset_num_layers: int = 1
    tb: TBConfig = TBConfig()
    moql: MOQLConfig = MOQLConfig()
    a2c: A2CConfig = A2CConfig()
    fm: FMConfig = FMConfig()
    sql: SQLConfig = SQLConfig()
