#!/usr/bin/env python3
"""
Testing of applying TTOI

"""

import numpy as np
import tensorly as tl
from tensorly import random
from tensorly.tt_tensor import tt_to_tensor
from tensorly.testing import assert_, assert_class_wrapper_correctly_passes_arguments
from tensorly.contrib.decomposition.tt_TTOI import tensor_train_OI, TensorTrain_OI


def test_TTOI(monkeypatch):
    """Test for the tt_TTOI function (Tensor train orthogonal iteration)"""
    rng = tl.check_random_state(1234)
    n_iter = 4
    tol = 1e-3

    # Exactly low-TT-rank tensors (no noise) must be recovered up to tolerance
    for order in range(3, 5):
        rank = (1,) + (3,) * (order - 1) + (1,)
        shape = (10,) * order
        true_tensor = random.random_tt(
            shape=shape, rank=rank, random_state=rng, full=True
        )

        # run TTOI
        factors_list, full_tensor_list, _ = tensor_train_OI(
            data_tensor=true_tensor,
            rank=rank,
            n_iter=n_iter,
            trajectory=True,
            return_errors=True,
        )

        # the TT factors of the last iteration have the requested ranks
        tt_factors = factors_list[-1]
        assert tuple(tl.shape(f)[0] for f in tt_factors) + (1,) == rank

        # the relative reconstruction error is below a fixed tolerance
        rec_error = tl.norm(tt_to_tensor(tt_factors) - true_tensor, 2) / tl.norm(
            true_tensor, 2
        )
        assert_(tl.to_numpy(rec_error) < tol)

        # and so is the one of every iteration
        for full_tensor in full_tensor_list:
            rel_error = tl.norm(full_tensor - true_tensor, 2) / tl.norm(true_tensor, 2)
            assert_(tl.to_numpy(rel_error) < tol)

    assert_class_wrapper_correctly_passes_arguments(
        monkeypatch, tensor_train_OI, TensorTrain_OI, ignore_args={}, rank=rank
    )
