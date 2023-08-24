from dataclasses import dataclass
from typing import Optional
    
@dataclass
class InsertionBufferConfig:
    strategy: str = "fifo"
    sim_thresh: float = 0.7 
    reward_thresh: float = 0.9 

@dataclass
class QuantileSamplingCOnfig:
    alpha: float = 0.1
    beta: float = 0.5

@dataclass
class WeightedSamplingConfig:
    reward_power: float = 1.0

@dataclass
class SamplingBufferConfig:
    strategy: str = "uniform"
    weighted: WeightedSamplingConfig = WeightedSamplingConfig()
    quantile: QuantileSamplingCOnfig = QuantileSamplingCOnfig()

@dataclass
class ReplayConfig:
    """Replay buffer configuration

    Attributes
    ----------
    use : bool
        Whether to use a replay buffer
    capacity : int
        The capacity of the replay buffer
    warmup : int
        The number of samples to collect before starting to sample from the replay buffer
    hindsight_ratio : float
        The ratio of hindsight samples within a batch
    insertion_strategy : str
        The strategy to use for inserting samples into the replay buffer
    diversity_thresh : float
        The diversity threshold for the insertion strategy "fifo_diversity_thresh"
    reward_thresh : float
        The reward threshold for the insertion strategy "fifo_reward_thresh"
    sampling_strategy : str
        The strategy to use for sampling from the replay buffer
    """

    use: bool = False
    capacity: Optional[int] = None
    warmup: Optional[int] = None
    hindsight_ratio: float = 0
    insertion: InsertionBufferConfig = InsertionBufferConfig()
    sampling: SamplingBufferConfig = SamplingBufferConfig()
