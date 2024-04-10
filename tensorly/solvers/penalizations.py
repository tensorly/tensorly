import tensorly as tl
import numpy as np
import copy
import warnings

# Weights processing
def process_regularization_weights(
    ridge_coefficients, sparsity_coefficients, n_modes, rescale=True
):
    """A utility function to process input sparisty and ridge coefficients, and return them in the correct format for decomposition functions.

    Parameters
    ----------
    ridge_coefficients : {list of floats, float, None}
        The ridge regularization parameter(s) to process
    sparsity_coefficients : {list of floats, float, None}
        The sparsity regularization parameter(s) to process
    n_modes : int
        number of modes in the decomposition which the coefficients are processed for.
    rescale : bool, optional
        Decides if the algorithm will use a rescaling strategy to minimize the regularization terms.
        Disabled in current version.
        Default True

    Returns
    -------
    list of floats
        ridge coefficients, processed
    list of floats
        sparsity coefficients, processed
    bool
        Is rebalance off or on
    list of int
        contains the homogeneity degrees of the regularization terms
    """
    if ridge_coefficients is None or isinstance(ridge_coefficients, (int, float)):
        # Populate None or the input float in a list for all modes
        ridge_coefficients = [ridge_coefficients] * n_modes
    if sparsity_coefficients is None or isinstance(sparsity_coefficients, (int, float)):
        # Populate None or the input float in a list for all modes
        sparsity_coefficients = [sparsity_coefficients] * n_modes
    # Convert None to 0
    for i in range(n_modes):
        if ridge_coefficients[i] == None:
            ridge_coefficients[i] = 0
        if sparsity_coefficients[i] == None:
            sparsity_coefficients[i] = 0

    # Add constant to check if rescale strategy is used
    if any(sparsity_coefficients) or any(ridge_coefficients):
        reg_is_used = True
    else:
        reg_is_used = False
    # adding ridge penalty unless done by user on non-sparse/ridge factors to avoid degeneracy
    if any(sparsity_coefficients):
        for i in range(n_modes):
            if abs(sparsity_coefficients[i]) + abs(ridge_coefficients[i]) == 0:
                warnings.warn(
                    f"Ridge coefficient set to max l1coeffs {max(sparsity_coefficients)} on mode {i} to avoid degeneracy"
                )
                ridge_coefficients[i] = max(sparsity_coefficients)

    # Avoid issues by printing warning when both l1 and l2 are imposed on the same mode
    disable_rebalance = True
    if rescale and reg_is_used:
        disable_rebalance = False
        for i in range(n_modes):
            if abs(sparsity_coefficients[i]) and abs(ridge_coefficients[i]):
                warnings.warn(
                    f"Optimal rebalancing strategy not designed for l1 and l2 simultaneously applied on the same mode. Removing rebalancing from algorithm."
                )
                disable_rebalance = True

    # homogeneity degrees of penalties
    hom_deg = None
    if not disable_rebalance:
        hom_deg = tl.tensor(
            [
                1.0 * (sparsity_coefficients[i] > 0) + 2.0 * (ridge_coefficients[i] > 0)
                for i in range(n_modes)
            ]
        )  # +1 for the core

    return ridge_coefficients, sparsity_coefficients, disable_rebalance, hom_deg


def scale_factors_fro(
    tensor,
    data,
    sparsity_coefficients,
    ridge_coefficients,
    format_tensor="cp",
    nonnegative=False,
):
    """
    Optimally scale Tucker tensor [G;A,B,C] in

    :math: `min_x \|data - x^{n_modes} [G;A_1,A_2,A_3]\|_F^2 + \sum_i sparsity_coefficients_i \|A_i\|_1 + \sum_j ridge_coefficients_j \|A_j\|_2^2`

    This avoids a scaling problem when starting the separation algorithm, which may lead to zero-locking.
    The problem is solved by finding the positive roots of a polynomial.

    Works with any number of modes and both CP and Tucker, as specified by the `format` input. For "tucker" format, sparsity and ridge have an additional final value for the core reg.

    note: sparsity works only under nonnegativity constraints
    note: comment on nonnegative keyword
    """
    factors = copy.deepcopy(tensor[1])
    if format_tensor == "tucker":
        factors.append(tensor[0])
    n_modes = len(factors)
    l1regs = [
        sparsity_coefficients[i] * tl.sum(tl.abs(factors[i])) for i in range(n_modes)
    ]
    l2regs = [ridge_coefficients[i] * tl.norm(factors[i]) ** 2 for i in range(n_modes)]
    # We define a polynomial
    # a x^{2n_modes} + b x^{n_modes} + c x^{2} + d x^{1}
    # and find the roots of its derivative, compute the value at each one, and return the optimal scale x and scaled factors.
    a = (tensor.norm() ** 2) / 2
    b = -tl.sum(data * tensor.to_tensor())
    c = sum(l2regs)
    d = sum(l1regs)
    poly = [0 for i in range(2 * n_modes + 1)]
    poly[1] = d
    poly[2] = c
    poly[n_modes] = b
    poly[2 * n_modes] = a
    poly.reverse()
    grad_poly = [0 for i in range(2 * n_modes)]
    grad_poly[0] = d
    grad_poly[1] = 2 * c
    grad_poly[n_modes - 1] = n_modes * b
    grad_poly[2 * n_modes - 1] = 2 * n_modes * a
    grad_poly.reverse()
    roots = np.roots(grad_poly)
    current_best = np.Inf
    best_x = 0
    for sol in roots:
        if sol.imag < 1e-16:
            sol = sol.real
            if sol > 0 or not nonnegative:
                val = np.polyval(poly, sol)
                if val < current_best:
                    current_best = val
                    best_x = sol
    if current_best == np.Inf:
        warnings.warn("No solution to optimal scaling")
        return tensor, None

    # We have the optimal scale
    for i in range(n_modes):
        factors[i] *= best_x

    if format_tensor == "tucker":
        return tl.tucker_tensor.TuckerTensor((factors[-1], factors[:-1])), best_x
    return tl.cp_tensor.CPTensor((None, factors)), best_x
