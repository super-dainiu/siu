import torchvision.models as tm
import timm.models as tmm
import torch

from metacol.passes.shape_prop import shape_prop_pass
from colossalai.fx.tracer.experimental import symbolic_trace


tm_models = [
    tm.vgg11, tm.resnet18, tm.densenet121, tm.mobilenet_v3_small, tm.resnext50_32x4d,
    tm.wide_resnet50_2, tm.regnet_x_16gf, tm.mnasnet0_5, tm.efficientnet_b0,
]

tmm_models = [
    tmm.resnest.resnest50d, tmm.beit.beit_base_patch16_224, tmm.cait.cait_s24_224, 
    # tmm.efficientnet.efficientnetv2_m, tmm.dpn.dpn68, tmm.densenet.densenet121, tmm.rexnet.rexnet_100,    # bad-case
    tmm.resmlp_12_224, tmm.vision_transformer.vit_base_patch16_224, tmm.deit_base_distilled_patch16_224,
    tmm.convnext.convnext_base, tmm.vgg.vgg11, tmm.swin_transformer.swin_base_patch4_window7_224
]

def _check_gm_validity(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        assert 'meta_data' in node.meta, f'meta_data not found in {node}'
        
def test_torchvision_shape_prop():
    for m in tm_models:
        model = m()
        data = torch.rand(100, 3, 224, 224, device='meta')
        meta_args = {
            "x": data,
        }
        gm = symbolic_trace(model, meta_args=meta_args)
        shape_prop_pass(gm, data, device=torch.device('cuda:0'))
        _check_gm_validity(gm)

def test_timm_shape_prop():
    for m in tmm_models:
        model = m()
        data = torch.rand(100, 3, 224, 224, device='meta')
        meta_args = {
            "x": data,
        }
        gm = symbolic_trace(model, meta_args=meta_args)
        shape_prop_pass(gm, data, device=torch.device('cuda:0'))
        _check_gm_validity(gm)


if __name__ == "__main__":
    test_timm_shape_prop()
    test_torchvision_shape_prop()

