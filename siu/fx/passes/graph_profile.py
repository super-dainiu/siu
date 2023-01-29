import torch
import torch.fx
from torch.autograd.graph import saved_tensors_hooks
from torch.fx import GraphModule

from siu.fx.node_util import MetaInfo


class sim_env(saved_tensors_hooks):

    def __init__(self):
        super().__init__(self.pack_hook, self.unpack_hook)
        self.cache = {}

    def pack_hook(self, tensor: torch.Tensor):
        self.cache[tensor.data_ptr()] = tensor.unwrap() if hasattr(tensor, 'unwrap') else tensor
        return tensor

    def unpack_hook(self, tensor):
        return tensor


class GraphProfile(torch.fx.Interpreter):
    _profileable = ['call_function', 'call_module', 'call_method']

    def __init__(self, module: GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        self.global_hook = sim_env()

    def run_node(self, n: torch.fx.Node):
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        n_info = MetaInfo(n)
        if n.op in self._profileable:
            inner_hook = sim_env()
            with self.global_hook, inner_hook:
                (
                    n_info.fwd_flop,
                    n_info.bwd_flop,
                    n_info.fwd_comm,
                    n_info.bwd_comm,
                ) = getattr(self, n.op)(n.target, args, kwargs)

    def fetch_initial_env(self):
        initial_env = {}
        for node in self.module.graph.nodes:
            initial_env[node] = node.meta['info'].output
        return initial_env

    def propagate(self, *args):
        initial_env = self.fetch_initial_env()
        return self.run(*args, initial_env=initial_env)
