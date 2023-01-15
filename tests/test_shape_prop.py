import timm.models as tmm
import torch
import torchvision.models as tm

from siu._subclasses import MetaTensorMode
from siu.fx.passes.shape_prop import shape_prop_pass, register_shape_impl
from siu.fx import symbolic_trace
from zoo import tm_models, tmm_models

def _check_gm_validity(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        assert node.meta['info'].data, f'In {gm.__class__.__name__}, {node} has no activation.'

@register_shape_impl(torch.nn.functional.linear)
def linear_impl(*args, **kwargs):
    print('siuuuu!')
    return torch.nn.functional.linear(*args, **kwargs)

def test_torchvision_shape_prop():
    for m in tm_models:
        with MetaTensorMode():
            model = m()
            data = torch.rand(100, 3, 224, 224)
        meta_args = {
            "x": data,
        }
        gm = symbolic_trace(model, meta_args=meta_args)
        shape_prop_pass(gm, data, device=torch.device('cuda:0'))
        _check_gm_validity(gm)


def test_timm_shape_prop():
    for m in tmm_models:
        with MetaTensorMode():
            model = m()
            data = torch.rand(100, 3, 224, 224)
        meta_args = {
            "x": data,
        }
        gm = symbolic_trace(model, meta_args=meta_args)
        shape_prop_pass(gm, data, device=torch.device('cuda:0'))
        _check_gm_validity(gm)


if __name__ == "__main__":
    test_torchvision_shape_prop()
    test_timm_shape_prop()
