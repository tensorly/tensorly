def _validate_contraction_modes(shape1, shape2, modes, batched_modes=False):
    """Takes in the contraction modes (for a tensordot) and validates them
    
    Parameters
    ----------
    modes : int or tuple[int] or (modes1, modes2)
    batched_modes : bool, default is False
    
    Returns
    -------
    modes1, modes2 : a list of modes for each contraction
    """
    if isinstance(modes, int):
        if batched_modes:
            modes1 = [modes]
            modes2 = [modes]
        else:
            modes1 = list(range(-modes, 0))
            modes2 = list(range(0, modes))
    else:
        try:
            modes1, modes2 = modes
        except ValueError:
            modes1 = modes
            modes2 = modes
    try:
        modes1 = list(modes1)
    except TypeError:
        modes1 = [modes1]
    try:
        modes2 = list(modes2)
    except TypeError:
        modes2 = [modes2]

    if len(modes1) != len(modes2):
        if batched_modes: 
            message = f'Both tensors must have the same number of batched modes'
        else:
            message = 'Both tensors must have the same number of modes to contract along. '
        raise ValueError(message + f'However, got modes={modes}, '
                         f' i.e. {len(modes1)} modes for tensor 1 and {len(modes2)} mode for tensor 2'
                         f'(modes1={modes1}, and modes2={modes2})')
    ndim1 = len(shape1)
    ndim2 = len(shape2)
    for i in range(len(modes1)):
        if shape1[modes1[i]] != shape2[modes2[i]]:
            if batched_modes:
                message = 'Batch-dimensions must have the same dimensions in both tensors but got'
            else:
                message = 'Contraction dimensions must have the same dimensions in both tensors but got'
            raise ValueError(message + f' mode {modes1[i]} of size {shape1[modes1[i]]} and '
                             f' mode {modes2[i]} of size {shape2[modes2[i]]}.')
        if modes1[i] < 0:
            modes1[i] += ndim1
        if modes2[i] < 0:
            modes2[i] += ndim2

    return modes1, modes2
