import torchvision.models as tm
import torchaudio.models as am
import timm.models as tmm
import torch
import torch.distributed as dist

from siu._subclasses import MetaTensor, MetaTensorMode

from model_list import tm_models, tmm_models


def test_meta_mode():
    loss = torch.sum
    for m in tm_models + tmm_models:
        with MetaTensorMode():
            model = m()
            loss(model(torch.rand(2, 3, 224, 224))).backward()


if __name__ == '__main__':
    test_meta_mode()