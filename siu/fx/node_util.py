from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Union

import torch
from torch.autograd.profiler_util import _format_memory, _format_time
from torch.fx import Graph, GraphModule, Node

from siu.envs import MeshConfig


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

    Usage:
        >>> for node in graph.nodes:
        >>>     n_info = MetaInfo(node)     # will create a new MetaInfo instance and store in node.meta['info']
        >>>                                 # if not exist, otherwise return the existing one
        >>>     n_info.data = ...   # set the data field

    Remarks:
        This feature is experimental and all the entries are subject to change.
    """

    # reference
    node: Node

    # should be updated after each graph manipulation
    # ============================== Update ====================================
    # parameter and buffer within ``Node``
    parameters: Dict[str, torch.nn.Parameter] = field(default_factory=dict)
    buffers: Dict[str, torch.Tensor] = field(default_factory=dict)

    # intermediate tensor as output
    data: Tuple[torch.Tensor] = ()

    # memory allocation
    saved_fwd_input: Tuple[torch.Tensor] = ()
    saved_fwd_buffer: Tuple[torch.Tensor] = ()    # [batchnorm (mean, var), relu (output), ...]
    saved_bwd_buffer: Tuple[torch.Tensor] = ()

    # compute cost
    fwd_flop: Optional[int] = 0
    bwd_flop: Optional[int] = 0

    # communication cost (should be the size in bytes of communication)
    fwd_comm: Optional[int] = 0
    bwd_comm: Optional[int] = 0

    # should keep the same whenever manipulated
    # ============================= Invariant ==================================
    to_recompute: Tuple[torch.Tensor] = ()    # (region_0, region_1, ...) support nested codegen
    to_offload: Optional[bool] = False
    sharding_spec: str = 'RR'

    def __new__(cls, node: Node, **kwargs):
        if node.meta.get('info', None) is not None:

            def _dummy(*args, **kwargs):
                pass

            cls.__init__ = _dummy
            return node.meta['info']
        return super().__new__(cls)

    def __post_init__(self):
        self.node.meta['info'] = self

    @property
    def fwd_time(self, tflops: float = MeshConfig.TFLOPS, bandwidth: float = MeshConfig.BANDWIDTH):
        return self.fwd_flop / tflops + self.fwd_comm / bandwidth

    @property
    def bwd_time(self, tflops: float = MeshConfig.TFLOPS, bandwidth: float = MeshConfig.BANDWIDTH):
        return self.bwd_flop / tflops + self.bwd_comm / bandwidth

    @property
    def param_size(self):
        return compute_size_in_bytes(self.parameters)

    @property
    def buffer_size(self):
        return compute_size_in_bytes(self.buffers)

    @property
    def size(self):
        return compute_size_in_bytes(self.data)

    def __repr__(self):
        s = f'Node {self.node.name}'
        if self.parameters:
            s += f'\n has parameter of size {_format_memory(self.param_size)}'
        if self.buffers:
            s += f'\n has buffer of size {_format_memory(self.buffer_size)}'
        if self.data:
            s += f'\n has activation of size {_format_memory(self.size)}'
        s += f'\n to_recompute = {self.to_recompute}'\
            f'\n to_offload = {self.to_offload}'\
            f'\n sharding_spec = {self.sharding_spec}'
        return s


def compute_size_in_bytes(elem: Union[torch.Tensor, Dict, List, Tuple, int]) -> int:
    """Compute the size of a tensor or a collection of tensors in bytes.

    Args:
        elem (Union[torch.Tensor, Dict, List, Tuple, int]): Arbitrary nested ``torch.Tensor`` data structure.

    Returns:
        int: The size of the tensor or the collection of tensors in bytes.
    """
    nbytes = 0
    if isinstance(elem, torch.Tensor):
        if elem.is_quantized:
            nbytes += elem.numel() * torch._empty_affine_quantized([], dtype=elem.dtype).element_size()
        else:
            nbytes += elem.numel() * torch.tensor([], dtype=elem.dtype).element_size()
    elif isinstance(elem, dict):
        value_list = [v for _, v in elem.items()]
        nbytes += compute_size_in_bytes(value_list)
    elif isinstance(elem, tuple) or isinstance(elem, list) or isinstance(elem, set):
        for e in elem:
            nbytes += compute_size_in_bytes(e)
    return nbytes
