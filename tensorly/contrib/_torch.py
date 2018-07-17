import torchvision.models as models
from .. import get_backend


def pytorch_model(model='resnet18'):
    """
    Parameters
    ----------
    models : str, optional. default ``inception``
        Other available models are any of the keys in
        https://pytorch.org/docs/stable/torchvision/models.html
        e.g., ``'alexnet'`` or ``vgg16``

    Returns
    -------
    params : list
        List of PyTorch tensors that represent the weights of a pretrained
        neural network
    """
    if get_backend() != 'pytorch':
        raise ValueError("This function is only implemented PyTorch backend. "
                         "Run tensorly.set_backend('pytorch') to prevent "
                         "this error")
    fn = getattr(models, model)
    net = fn(pretrained=True)
    return list(net.parameters())
