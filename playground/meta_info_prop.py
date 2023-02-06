import torch
import torchvision.models as tm
from colossalai.fx.passes import ConcreteInfoProp, MetaInfoProp
from colossalai.fx.profiler import calculate_fwd_out, calculate_fwd_tmp, is_compatible_with_meta, parameter_size
from torch.autograd.profiler_util import _format_memory

from siu.fx import symbolic_trace
from siu.fx.passes.graph_profile import sim_env


def extract_forward_mem(gm: torch.fx.GraphModule):
    node_size = 0
    param_size = 0
    for node in gm.graph.nodes:
        node_size += calculate_fwd_tmp(node)
        node_size += calculate_fwd_out(node)
    param_size = parameter_size(gm)
    return node_size, param_size


mod = tm.mobilenet_v2()
data = torch.rand(8, 3, 224, 224)
meta_args = {
    "x": data,
}
gm = symbolic_trace(mod, meta_args=meta_args)
interp = MetaInfoProp(gm)
interp.propagate(data)
# print(interp.summary())
activation_size, param_size = extract_forward_mem(gm)
print(f"activation_size: {_format_memory(activation_size)}")

from make_fx import make_fx


class Mod(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.silu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        x = x + 0
        x = self.silu(x)
        return x


env = sim_env()
with env:
    graph = make_fx(Mod(), torch.rand(8, 3, 224, 224, requires_grad=True).cuda())
    print(graph.python_code('self').src)
    print(len(env.ctx.keys()))

graph = make_fx(torch.nn.SiLU(inplace=False), torch.rand(8, 3, 224, 224, requires_grad=True).cuda())
print(graph.python_code('self').src)
