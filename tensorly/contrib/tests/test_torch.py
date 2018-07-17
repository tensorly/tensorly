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
    with pytest.raises(AttributeError, match="no attribute 'junk'"):
        contrib.pytorch_model(model='junk')
