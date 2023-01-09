from dataclasses import dataclass
from typing import Optional, Callable, List, Union, Tuple, Dict

import torch

from ..utils import compute_size_in_bytes
from .profiler import meta_info_fn


def _flop2time(flop: int, tflops: float) -> float:
    return flop / tflops
    
def _comm2time(comm_size: int, bandwidth: float) -> float:
    return comm_size / bandwidth
    
def _estimate_time_with_spec(flop: int, sharding_spec: str, tflops) -> float:
    # process sharding spec (TODO)
    processed_flop = flop
    return _flop2time(processed_flop, tflops)

@dataclass
class MetaInfo:
    r"""
    The base class to store all profiling and static graph analysis information
    needed for auto-parallel system in Colossal-AI.
    ============================================================================
                            -------------------------------
                            |          FX.Node            |
    [fwd_input] are    ---> | [fwd_in]          [bwd_buf] |    <-----
    placeholders saved for  |     | \__________     |     |
    backward.               |     |            \    |     |
                            | [fwd_buf] ------> [bwd_buf] |    <-----
                            |     |  \_________     |     |    [bwd_buffer] marks the peak
                            |    / \           \    |     |    memory in backward pass.
    [x] is not counted ---> | [x]  [fwd_buf] -> [bwd_buf] |    <-----
    in [fwd_buffer] because |          |  \_____    |     |
    it is not saved for     |          |        \   |     |
    backward.               |      [activat]     \  |     |    <----- [activation] is potentially 
                            -------------------------------    [fwd_input] for the next node.
    ============================================================================
    """
    
    # reference
    node: torch.fx.Node
    
    # parameter within ``Node``
    parameter: List[torch.nn.Parameter] = []
    
    # intermediate tensor as output
    activation: List[torch.Tensor] = []
    
    # memory allocation
    saved_fwd_input: List[torch.Tensor] = []
    saved_fwd_buffer: List[torch.Tensor] = []    # [batchnorm (mean, var), relu (output), ...]
    saved_bwd_buffer: List[torch.Tensor] = []
    
    # compute cost
    fwd_flop: Optional[int] = 0
    bwd_flop: Optional[int] = 0
    
    # communication cost (should be the size in bytes of communication)
    fwd_comm: Optional[int] = 0
    bwd_comm: Optional[int] = 0
    
    # recompute
    to_recompute: Optional[bool] = False
    
    # offload
    to_offload: Optional[bool] = False
    
    # sharding spec
    sharding_spec: str = 'RR'
    
    def __post_init__(self):
        self.saved_fwd_input, self.saved_fwd_buffer, self.saved_bwd_buffer, \
            self.fwd_flops, self.bwd_flops, self.fwd_comm, self.bwd_comm = \
            meta_info_fn[self.node.target](self.node)

    @property
    def fwd_time(self, tflops: float = None, bandwidth: float = None):
        return _estimate_time_with_spec(self.fwd_flop, self.sharding_spec, tflops) + _comm2time(self.fwd_comm, bandwidth)
    
    @property
    def bwd_time(self, tflops: float = None, bandwidth: float = None):
        return _estimate_time_with_spec(self.bwd_flop, self.sharding_spec, tflops) + _comm2time(self.bwd_comm, bandwidth)