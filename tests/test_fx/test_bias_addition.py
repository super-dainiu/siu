import torch
from torch.utils.checkpoint import checkpoint

from siu.fx import symbolic_trace


class LinearModel(torch.nn.Module):

    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = self.linear(x)
        x = x * 2
        return x


class ConvModel(torch.nn.Module):

    def __init__(self, in_channel, out_channels, kernel_size, bias) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channel,
                                    out_channels,
                                    kernel_size,
                                    bias=bias,
                                    padding=1,
                                    stride=2,
                                    dilation=2,
                                    groups=3)

    def forward(self, x):
        x = self.conv(x)
        x = x * 2
        return x


class SiuModel(torch.nn.Module):

    def __init__(self, bias) -> None:
        super().__init__()
        self.linear = LinearModel(3, 3, bias)
        self.conv = ConvModel(3, 6, 3, bias)

    def forward(self, x):
        x = self.linear(x)
        x = checkpoint(self.conv, x)
        return x


def test_siu_model():
    model = SiuModel(bias=True)
    model(torch.rand(3, 3, 3))
    gm = symbolic_trace(model, meta_args={'x': torch.rand(3, 3, 3)}, trace_act_ckpt=True, bias_addition_split=True)
    print(gm.code)


if __name__ == '__main__':
    test_siu_model()
