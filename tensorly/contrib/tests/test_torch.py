import pytest
import tensorly as tl
from ... import contrib
from ... import set_backend, get_backend


def test_pytorch_model():
    if tl._BACKEND == 'pytorch':
        import torch
        with pytest.raises(AttributeError, match="no attribute 'junk'"):
            contrib.pytorch_model(model='junk')
    else:
        with pytest.raises(ValueError, match="set_backend"):
            contrib.pytorch_model(model='resnet18')
