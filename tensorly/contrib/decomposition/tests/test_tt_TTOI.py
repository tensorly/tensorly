#!/usr/bin/env python3
"""
Testing of applying TTOI

"""

import numpy as np
import tensorly as tl
from tensorly import random
from tensorly.testing import assert_, assert_class_wrapper_correctly_passes_arguments
from tensorly.contrib.decomposition.tt_TTOI import tensor_train_OI, TensorTrain_OI


def test_TTOI(monkeypatch):
    """Test for the tt_TTOI function (Tensor train orthogonal iteration)"""
    rng = tl.check_random_state(1234)
    n_iter = 4

    # Generate tensor true_tensor with low tensor train rank, and its noisy observation data_tensor
    for i in range(3, 5):
        rank = tuple(np.ones(i + 1).astype(int))
        shape = tuple(np.ones(i).astype(int) * 20)
        true_tensor = random.random_tt(
            shape=shape, rank=rank, random_state=rng, full=True
        )
        noise_tensor = tl.tensor(rng.standard_normal(shape), **tl.context(true_tensor))
        data_tensor = true_tensor + noise_tensor

        # run TTOI
        _, full_tensor_list, approx_errors = tensor_train_OI(
            data_tensor=data_tensor,
            rank=rank,
            n_iter=n_iter,
            trajectory=True,
            return_errors=True,
        )

        # Check that the approximation error monotonically decreases
        tensor_norm = tl.norm(data_tensor, 2)
        for i, error in enumerate(approx_errors):
            if i:
                assert (tl.to_numpy((previous_error - error) / tensor_norm)) <= 2e-3
            previous_error = error
        # assert (np.all(np.diff(tl.to_numpy(approx_errors)) <= 1e-3))

        # check that the estimation error of TTOI improves from initialization (TTSVD)
        estimation_errors = [
            tl.norm(full_tensor_list[i] - true_tensor, 2) / tl.norm(true_tensor, 2)
            for i in range(n_iter)
        ]
        assert_(
            tl.to_numpy(estimation_errors[0] - estimation_errors[n_iter - 1]) >= 2e-3
        )

    assert_class_wrapper_correctly_passes_arguments(
        monkeypatch, tensor_train_OI, TensorTrain_OI, ignore_args={}, rank=rank
    )
