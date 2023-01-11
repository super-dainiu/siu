from dataclasses import dataclass, field
import warnings
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Union

import torch
from torch.fx import Graph, GraphModule, Node
from torch.autograd.profiler_util import _format_memory, _format_time

from siu.envs import MeshConfig
from siu.utils import compute_size_in_bytes


def _flop_to_time(flop: int, tflops: float) -> float:
    return flop / tflops


def _comm_to_time(comm_size: int, bandwidth: float) -> float:
    return comm_size / bandwidth


def _estimate_time_with_spec(flop: int, tflops: float, sharding_spec: str = None) -> float:
    # process sharding spec (TODO: some-man)
    processed_flop = flop
    return _flop_to_time(processed_flop, tflops)


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

    # reference
    node: Node

    # should be updated after each graph manipulation
    # ============================== Update ====================================
    # parameter within ``Node``
    parameter: Tuple[torch.nn.Parameter] = ()

    # intermediate tensor as output
    activation: Tuple[torch.Tensor] = ()

    # memory allocation
    saved_fwd_input: Tuple[torch.Tensor] = ()
    saved_fwd_buffer: Tuple[torch.Tensor] = ()   # [batchnorm (mean, var), relu (output), ...]
    saved_bwd_buffer: Tuple[torch.Tensor] = ()

    # compute cost
    fwd_flop: Optional[int] = 0
    bwd_flop: Optional[int] = 0

    # communication cost (should be the size in bytes of communication)
    fwd_comm: Optional[int] = 0
    bwd_comm: Optional[int] = 0

    # should keep the same whenever manipulated
    # ============================= Invariant ==================================
    to_recompute: Tuple[torch.Tensor] = ()  # (region_0, region_1, ...) support nested codegen
    to_offload: Optional[bool] = False
    sharding_spec: str = 'RR'

    def __new__(self, node: Node, **kwargs):
        if node.meta.get('info', None) is not None:
            return node.meta['info']
        return super().__new__(self)

    def __post_init__(self):
        self.node.meta['info'] = self

    @property
    def fwd_time(self, tflops: float = MeshConfig.TFLOPS, bandwidth: float = MeshConfig.BANDWIDTH):
        return _estimate_time_with_spec(self.fwd_flop, tflops, self.sharding_spec) + _comm_to_time(
            self.fwd_comm, bandwidth)

    @property
    def bwd_time(self, tflops: float = MeshConfig.TFLOPS, bandwidth: float = MeshConfig.BANDWIDTH):
        return _estimate_time_with_spec(self.bwd_flop, tflops, self.sharding_spec) + _comm_to_time(
            self.bwd_comm, bandwidth)

    @property
    def param_size(self):
        # TODO: specify sharding spec
        return compute_size_in_bytes(self.parameter)

    @property
    def activ_size(self):
        return compute_size_in_bytes(self.activation)

    def __repr__(self):
        s = f'Node {self.node.name}'
        if self.parameter:
            s += f'\n has parameter of size {_format_memory(self.param_size)}'
        if self.activation:
            s += f'\n activation {_format_memory(self.activ_size)}'
        s += f'\n to_recompute {self.to_recompute}'\
            f'\n to_offload {self.to_offload}'\
                f'\n sharding_spec {self.sharding_spec}'
        return s
        

