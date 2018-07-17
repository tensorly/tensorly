import torch
import pytest
from ... import contrib
from ... import set_backend, get_backend


def test_pytorch_model_raises():
    assert get_backend() == 'numpy'
    with pytest.raises(ValueError, match="set_backend"):
        contrib.pytorch_model(model='resnet18')


def test_pytorch_model():
    set_backend('pytorch')
    params = contrib.pytorch_model(model='resnet18')
    shapes = [p.shape for p in params]
    set(shapes) == {torch.Size([64, 3, 7, 7]),
                    torch.Size([64]),
                    torch.Size([64, 64, 3, 3]),
                    torch.Size([128, 64, 3, 3]),
                    torch.Size([128, 64, 1, 1]),
                    torch.Size([128, 128, 3, 3]),
                    torch.Size([256, 128, 3, 3]),
                    torch.Size([256]),
                    torch.Size([128]),
                    torch.Size([256, 256, 3, 3]),
                    torch.Size([256, 128, 1, 1]),
                    torch.Size([512]),
                    torch.Size([1000, 512]),
                    torch.Size([1000]),
                    torch.Size([512, 256, 3, 3]),
                    torch.Size([512, 512, 3, 3]),
                    torch.Size([512, 256, 1, 1])}
