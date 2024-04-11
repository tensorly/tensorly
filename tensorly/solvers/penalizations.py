import tensorly as tl
import numpy as np
import copy
import warnings


# Weights processing
def process_regularization_weights(ridge_coefficients, sparsity_coefficients, n_modes):
    """A utility function to process input sparisty and ridge coefficients, and return them in the correct format for decomposition functions.

    Parameters
    ----------
    ridge_coefficients : {list of floats, float, None}
        The ridge regularization parameter(s) to process
    sparsity_coefficients : {list of floats, float, None}
        The sparsity regularization parameter(s) to process
    n_modes : int
        number of modes in the decomposition which the coefficients are processed for.

    Returns
    -------
    list of floats
        ridge coefficients, processed
    list of floats
        sparsity coefficients, processed
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

    # adding ridge penalty unless done by user on non-sparse/ridge factors to avoid degeneracy
    if any(sparsity_coefficients):
        for i in range(n_modes):
            if abs(sparsity_coefficients[i]) + abs(ridge_coefficients[i]) == 0:
                warnings.warn(
                    f"Ridge coefficient set to max l1coeffs {max(sparsity_coefficients)} on mode {i} to avoid degeneracy"
                )
                ridge_coefficients[i] = max(sparsity_coefficients)

    return ridge_coefficients, sparsity_coefficients
