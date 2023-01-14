import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from siu.fx import symbolic_trace, MetaInfo


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 10)
        self.b = nn.Linear(10, 10)
        self.c = nn.Linear(10, 10)
        self.d = nn.Linear(10, 10)
        self.e = nn.Linear(10, 10)

    def ckpt_0_0_0_0(self, x):
        return self.b(x)
    
    def ckpt_0_0_0(self, x):
        return self.a(x) + checkpoint(self.ckpt_0_0_0_0, x)

    def ckpt_0_0_1(self, x):
        return self.b(x) + self.c(x)

    def ckpt_0_0(self, x, y):
        return checkpoint(self.ckpt_0_0_0, x) + checkpoint(self.ckpt_0_0_1, y)

    def ckpt_0_1(self, x):
        return self.d(x)

    def ckpt_0(self, x, y):
        return checkpoint(self.ckpt_0_0, x, y) + checkpoint(self.ckpt_0_1, x) + self.e(y)

    def forward(self, x):
        return checkpoint(self.ckpt_0, x, x)


if __name__ == "__main__":
    model = MyModule()
    x = torch.rand(10, 10)
    gm = symbolic_trace(model, meta_args={'x': x}, trace_act_ckpt=True)
    for node in gm.graph.nodes:
        print(node.meta)
    print(gm.code)