import tensorly as tl
import numpy as np

# Author : Jeremy Cohen


def least_squares_nway(input_tensor, input_factors, update_mode):
    """ Least squares update along fixed mode update_mode

    Update the mode update_mode by solving a least squares problem. This is a
    first naive implementation to demonstrate the syntax of an optimization
    submodule.

    This function is strictly superior to a least squares solver ran on the
    matricized problem min_X ||Y - AX||_F^2 since A is structured as a
    Kronecker product of other factors.

    Parameters
    ----------
    input_tensor : tensor of arbitrary order. If provided tensor is second
    order, then it is supposed to be already unfolded. Otherwise, it is
    unfolded along mode update_mode.
    input_factors : current estimates for the PARAFAC decomposition of
    input_tensor. The value of input_factor[update_mode] will be updated using
    a least squares update.
    update_mode : the mode to be updated.

    Returns -------
    out_factors : same as input_factors, except on mode
    update_mode where a least squares rule is used to compute a new estimated
    factor.
    out_cross_products : A^t*Y and A^t*A. Useful for instance for
    error computation, since evaluating the cost function requires such
    products.

    """

    # Compute Unfolding if required.

    # Computing cross-products

    # Update using backend linear system solver
