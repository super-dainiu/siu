import pytest
import torch
import torchvision.models as tm
from zoo import tm_models, tmm_models

from siu._subclasses import MetaTensorMode, flop_count


@pytest.mark.parametrize('m', tm_models + tmm_models)
def test_flop_count(m):
    x = torch.rand(2, 3, 224, 224)
    with MetaTensorMode():    # save time for testing
        module = m()
    rs = flop_count(module, x, verbose=True)
    assert rs > 0, f'flop count of {m.__name__} is {rs}'


if __name__ == '__main__':
    test_flop_count(tm.resnet18, torch.rand(2, 3, 224, 224))
