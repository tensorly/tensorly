import tensorly as tl
import numpy as np
from tensorly.base import unfold
from tensorly.tenalg import khatri_rao

# Author : Jeremy Cohen


def least_squares_nway(input_tensor, input_factors,
                       rank, norm_tensor, fixed_modes):
    """ One pass of Alternating Least squares update along all modes

    Update the factors by solving a least squares problem per mode. This is a
    first naive implementation to demonstrate the syntax of an optimization
    submodule.

    This function is strictly superior to a least squares solver ran on the
    matricized problems min_X ||Y - AX||_F^2 since A is structured as a
    Kronecker product of other factors.

    Parameters
    ----------
    input_tensor : tensor of arbitrary order. 
    input_factors : current estimates for the PARAFAC decomposition of
    input_tensor. The value of input_factor[update_mode] will be updated using
    a least squares update.
    rank : rank of the decomposition.
    norm_tensor : the Frobenius norm of the input tensor
    fixed_modes : indexes of modes that are not updated

    Returns -------
    out_factors : updated inputs factors
    rec_error : residual error after the ALS steps.
    """
  
    # Generating the mode update sequence
    gen = (mode for mode in range(tl.ndim(input_tensor)) if mode not in fixed_modes)

    #  for mode in range(tl.ndim(input_tensor)):
    for mode in gen:

        # Unfolding
        unfoldY = unfold(input_tensor,mode)

        # Computing Hadamard of cross-products
        cross = tl.tensor(np.ones((rank, rank)), **tl.context(input_tensor))
        for i, factor in enumerate(input_factors):
            if i != mode:
                cross = cross*tl.dot(tl.transpose(factor),factor)

        # Computing the Khatri Rao product
        krao = khatri_rao(input_factors,skip_matrix=mode)
        rhs = tl.dot(unfoldY,krao)

        # Update using backend linear system solver
        input_factors[mode] = tl.transpose(
                tl.solve(tl.transpose(cross),tl.transpose(rhs)))

    # error computation (improved using precomputed quantities)
    rec_error = norm_tensor ** 2 - 2*tl.dot(
            tl.tensor_to_vec(factor),tl.tensor_to_vec(
                rhs)) + tl.norm(tl.dot(factor,tl.transpose(krao)),2)**2
    rec_error = rec_error ** (1/2) / norm_tensor

    # outputs
    return input_factors, rec_error

def nn_least_squares_nway(input_tensor, input_factors, rank, norm_tensor):
    """ One pass of Nonnegative Alternating Least squares update along all modes
    Implements a projected ALS which is often BAD, this is just for demo

    Update the factors by solving a least squares problem per mode. This is a
    first naive implementation to demonstrate the syntax of an optimization
    submodule.

    This function is strictly superior to a least squares solver ran on the
    matricized problems min_X ||Y - AX||_F^2 since A is structured as a
    Kronecker product of other factors.

    Parameters
    ----------
    input_tensor : tensor of arbitrary order. 
    input_factors : current estimates for the PARAFAC decomposition of
    input_tensor. The value of input_factor[update_mode] will be updated using
    a projected least squares update.
    rank : rank of the decomposition.
    norm_tensor : the Frobenius norm of the input tensor

    Returns -------
    out_factors : updated inputs factors
    rec_error : residual error after the projected ALS steps.
    """

    for mode in range(tl.ndim(input_tensor)):

        # Unfolding
        unfoldY = unfold(input_tensor,mode)

        # Computing Hadamard of cross-products
        cross = tl.tensor(np.ones((rank, rank)), **tl.context(input_tensor))
        for i, factor in enumerate(input_factors):
            if i != mode:
                cross = cross*tl.dot(tl.transpose(factor),factor)

        # Computing the Khatri Rao product
        krao = khatri_rao(input_factors,skip_matrix=mode)
        rhs = tl.dot(unfoldY,krao)

        # Update using backend linear system solver
        input_factors[mode] = tl.transpose(
                tl.solve(tl.transpose(cross),tl.transpose(rhs)))
        
        # Projection on the nonnegative orthant
        input_factors[mode][input_factors[mode]<0] = 0

    # error computation (improved using precomputed quantities)
    rec_error = norm_tensor ** 2 - 2*tl.dot(
            tl.tensor_to_vec(factor),tl.tensor_to_vec(
                rhs)) + tl.norm(tl.dot(factor,tl.transpose(krao)),2)**2
    rec_error = rec_error ** (1/2) / norm_tensor

    # outputs
    return input_factors, rec_error
