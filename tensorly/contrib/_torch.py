import torchvision.models as models


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
    fn = getattr(models, model)
    net = fn(pretrained=True)
    return list(net.parameters())
