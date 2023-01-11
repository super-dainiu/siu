import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from siu.fx import symbolic_trace


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(10, 10)
        self.layer_2 = nn.Linear(10, 10)
        self.layer_3 = nn.Linear(10, 10)
        self.layer_4 = nn.Linear(10, 10)
        self.layer_5 = nn.Linear(10, 10)
    
    def ckpt_0_0(self, x):
        return self.layer_1(x) + checkpoint(self.layer_2, x)

    def ckpt_0_1(self, x):
        return self.layer_2(x) + self.layer_3(x)

    def ckpt_0(self, x, y):
        return checkpoint(self.ckpt_0_0, x) + checkpoint(self.ckpt_0_1, y)

    def ckpt_1(self, x, y):
        return checkpoint(self.ckpt_0, x, y) + checkpoint(self.layer_4, x) + self.layer_5(y)

    def forward(self, x):
        return checkpoint(self.ckpt_1, x, x)


if __name__ == "__main__":
    model = MyModule()
    x = torch.rand(10, 10)
    gm = symbolic_trace(model, meta_args={'x': x}, trace_act_ckpt=True)
    for node in gm.graph.nodes:
        print(node.meta)