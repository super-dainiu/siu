from dataclasses import dataclass
from typing import Optional, Callable, List, Union, Tuple, Dict, ClassVar

import torch
from torch.fx import Node, Graph, GraphModule

from ..utils import compute_size_in_bytes
from ..envs import MeshConfig
from .profiler import meta_info_fn


def _flop2time(flop: int, tflops: float) -> float:
    return flop / tflops
    
def _comm2time(comm_size: int, bandwidth: float) -> float:
    return comm_size / bandwidth
    
def _estimate_time_with_spec(flop: int, tflops: float, sharding_spec: str = None) -> float:
    # process sharding spec (TODO: some-man)
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
    [fwd_input] are    ---> | [fwd_inp]         [bwd_buf] |    <-----
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
    # class variable: all registered functions to trace 'meta_info'
    _meta_info_func: ClassVar[Dict] = {}
    
    # reference
    node: Node
    module: Optional[GraphModule] = None 
    
    # parameter within ``Node``
    has_param: Optional[bool] = False
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
        self.module = node.graph.owning_module()

        node = self.node
        args = [arg.meta['meta_data'] for arg in node.args]
        self.saved_fwd_input, self.saved_fwd_buffer, self.saved_bwd_buffer, \
            self.fwd_flops, self.bwd_flops, self.fwd_comm, self.bwd_comm = \
            MetaInfo._meta_info_func[node.target](*args)

    @property
    def fwd_time(self, tflops: float = MeshConfig.TFLOPS, bandwidth: float = MeshConfig.BANDWIDTH):
        return _estimate_time_with_spec(self.fwd_flop, tflops, self.sharding_spec) + _comm2time(self.fwd_comm, bandwidth)
    
    @property
    def bwd_time(self, tflops: float = MeshConfig.TFLOPS, bandwidth: float = MeshConfig.BANDWIDTH):
        return _estimate_time_with_spec(self.bwd_flop, tflops, self.sharding_spec) + _comm2time(self.bwd_comm, bandwidth)

    @property
    def param_size(self):
        # TODO: specify sharding spec
        return compute_size_in_bytes(self.parameter)