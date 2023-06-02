import tensorly as tl
import numpy as np
import copy
import warnings

# Weights processing
def process_regularization_weights(ridge_coefficients, sparsity_coefficients, n_modes, rescale=True, pop_l2=False):
    if ridge_coefficients is None or isinstance(ridge_coefficients, (int, float)):
        # Populate None or the input float in a list for all modes
        ridge_coefficients = [ridge_coefficients] * n_modes
    if sparsity_coefficients is None or isinstance(sparsity_coefficients, (int, float)):
        # Populate None or the input float in a list for all modes
        sparsity_coefficients = [sparsity_coefficients] * n_modes 
    # TODO: check penalisation length
    # Convert None to 0
    for i in range(n_modes):
        if ridge_coefficients[i]==None:
            ridge_coefficients[i]=0
        if sparsity_coefficients[i]==None:
            sparsity_coefficients[i]=0

    # Add constant to check if rescale strategy is used
    if (any(sparsity_coefficients) or any(ridge_coefficients)):
        reg_is_used = True
    else:
        reg_is_used = False
    # adding ridge penalty unless done by user on non-sparse/ridge factors to avoid degeneracy
    # TODO add in doc; pop_l2 option to disable this behavior
    if not pop_l2 and any(sparsity_coefficients):
        for i in range(n_modes):
            if abs(sparsity_coefficients[i]) + abs(ridge_coefficients[i])==0:
                warnings.warn(f"Ridge coefficient set to max l1coeffs {tl.max(sparsity_coefficients)} on mode {i} to avoid degeneracy")
                ridge_coefficients[i] = tl.max(sparsity_coefficients)
    
    # Avoid issues by printing warning when both l1 and l2 are imposed on the same mode
    disable_rebalance = True
    if rescale and reg_is_used:
        disable_rebalance = False
        for i in range(n_modes):
            if abs(sparsity_coefficients[i]) and abs(ridge_coefficients[i]):
                warnings.warn(f"Optimal rebalancing strategy not designed for l1 and l2 simultaneously applied on the same mode. Removing rebalancing from algorithm.")
                disable_rebalance = True
    
    # homogeneity degrees of penalties
    hom_deg = None
    if not disable_rebalance:
        hom_deg = tl.tensor([1.0*(sparsity_coefficients[i]>0) + 2.0*(ridge_coefficients[i]>0) for i in range(n_modes)]) # +1 for the core

    return ridge_coefficients, sparsity_coefficients, reg_is_used, disable_rebalance, hom_deg

# CP specific
def cp_opt_balance(regs, hom_deg):
    '''
    Computes the multiplicative constants to scale factors columnwise such that regularizations are balanced.
    The problem solved is 
        min_{a_i} \sum a_i s.t.  \prod a_i^{p_i}=q
    where a_i = regs[i] and p_i = hom_deg[i]

    Parameters
    ----------
    regs: 1d np array
        the input regularization values
    hom_deg: 1d numpy array
        homogeneity degrees of each regularization term

    Returns
    -------
    scales: list of floats
        the scale to balance the factors. Its product should be one (scale invariance).
    '''
    # 0. If reg is zero, all factors should be zero
    if tl.prod(regs)==0:
        # TODO warning
        #print(f"zero rebalancing because regularization is null")
        return [0 for i in range(len(regs))]

    # 1. compute q
    prod_q = tl.prod(regs**(1/hom_deg))

    # 2. compute beta
    beta = (prod_q*tl.prod(hom_deg**(1/hom_deg)))**(1/tl.sum(1/hom_deg))

    # 3. compute scales
    scales = [(beta/regs[i]/hom_deg[i])**(1/hom_deg[i]) for i in range(len(regs))]

    return scales

def tucker_implicit_sinkhorn_balancing(factors, core, regs, lamb_g, hom_reg, itermax=10, verbose=False):
    """A one liner to balance factors and core using adaptive sinkhorn.
    Solves the regularization scaling problem, similar to Sinkhorn but with unknown marginals learnt on the fly.

    Parameters
    ----------
    factors : list of arrays, required
        factors of the Tucker model
    core : tl tensor, required 
        core of the Tucker model
    regs : list of floats, required
        the list of regularization values for the input factors. May contain zeroes.
    hom_reg : list of ints, required
        homogeneity degrees for the factors regularizations, and the core
    lamb_g : float, required
        regularization parameter for the core penalization
    hom_g : int, optional
        homogeneity degree for the core regularization, by default 1
    itermax : int, optional
        maximal number of scaling iterations, by default 10

    Returns
    -------
    factors, core : scaled factors and core
    scales : list of lists with the scaling of the core/factors on each mode
    """

    hom_g = hom_reg[-1]
    if hom_g==1:
        reg_g = lambda x: tl.abs(x)
    elif hom_g==2:
        reg_g = lambda x: x**2
    else:
        print("hom_g not 1 or 2 not implemented")

    # Precomputations
    dims = tl.shape(core)
    nmodes = core.ndim

    # initial marginals
    beta = copy.deepcopy(regs) # marginals of reg_g(tensor)

    for it in range(itermax):
        for mode in range(nmodes):
            core_marginal_mode = lamb_g*tl.sum(reg_g(core), axis=tuple([i for i in range(nmodes) if i!=mode]), keepdims=True)
            beta[mode] = (
                (hom_reg[mode]*beta[mode])**(1/hom_reg[mode])*
                (hom_g*core_marginal_mode.reshape(dims[mode]))**(1/hom_g)
                )**(1/(1/hom_g+1/hom_reg[mode]))
            # Marginals will be zero if one of fac or core has zero marginal. We put ones where there are zeros in the tensor marginals to have 0/1=0.
            core_marginal_mode[core_marginal_mode==0] = 1
            core *=(beta[mode].reshape(core_marginal_mode.shape)/core_marginal_mode/hom_g)**(1/hom_g)
        # some cost function
        if verbose:
            loss = sum([sum(marg) for marg in beta]) + lamb_g*tl.sum(reg_g(core))
            print(f"iteration {it} loss {loss}")
            #print(f"consistency {np.std([sum(mar) for mar in beta])}")

    for mode in range(nmodes):
        for q in range(factors[mode].shape[1]):
            if regs[mode][q]:
                factors[mode][:,q] *=  (beta[mode][q]/regs[mode][q]/hom_reg[mode])**(1/hom_reg[mode])
            else:
                factors[mode][:,q] = 0

    #print(f"G scales {scales_g},\nfac scales {scales}\nproduct {[scales_g[i]*scales[i] for i in range(nmodes)]}")
    return factors, core

def tucker_implicit_scalar_balancing(factors, core, regs, hom_deg):
    """A one liner to balance factors and core using scalar scaling.

    Parameters
    ----------
    factors : _type_
        _description_
    core : _type_
        _description_
    regs : _type_
        _description_
    hom_deg : _type_
        _description_
    """    """
    """
    scales = cp_opt_balance(np.array(regs),np.array(hom_deg))
    for mode in range(tl.ndim(core)):
        factors[mode] *= scales[mode]
    core = core*scales[-1]

    return factors, core, scales

def scale_factors_fro(tensor,data,sparsity_coefficients,ridge_coefficients, format_tensor="cp"):
    '''
    Optimally scale [G;A,B,C] in 
    
    min_x \|data - x^{n_modes} [G;A_1,A_2,A_3]\|_F^2 + \sum_i sparsity_coefficients_i \|A_i\|_1 + \sum_j ridge_coefficients_j \|A_j\|_2^2

    This avoids a scaling problem when starting the separation algorithm, which may lead to zero-locking.
    The problem is solved by finding the positive roots of a polynomial.

    Works with any number of modes and both CP and Tucker, as specified by the `format` input. For "tucker" format, sparsity and ridge have an additional final value for the core reg.
    '''
    factors = copy.deepcopy(tensor[1])
    if format_tensor=="tucker":
        factors.append(tensor[0])
    n_modes = len(factors)
    l1regs = [sparsity_coefficients[i]*tl.sum(tl.abs(factors[i])) for i in range(n_modes)]
    l2regs= [ridge_coefficients[i]*tl.norm(factors[i])**2 for i in range(n_modes)]
    # We define a polynomial
    # a x^{2n_modes} + b x^{n_modes} + c x^{2} + d x^{1}
    # and find the roots of its derivative, compute the value at each one, and return the optimal scale x and scaled factors.
    a = (tensor.norm()**2)/2
    b = -tl.sum(data*tensor.to_tensor())
    c = sum(l2regs)
    d = sum(l1regs)
    poly = [0 for i in range(2*n_modes+1)]
    poly[1] = d
    poly[2] = c
    poly[n_modes] = b
    poly[2*n_modes] = a
    poly.reverse()
    grad_poly = [0 for i in range(2*n_modes)]
    grad_poly[0] = d
    grad_poly[1] = 2*c
    grad_poly[n_modes-1] = n_modes*b
    grad_poly[2*n_modes-1] = 2*n_modes*a
    grad_poly.reverse()
    roots = np.roots(grad_poly)
    current_best = np.Inf
    best_x = 0
    for sol in roots:
        if sol.imag<1e-16:
            sol = sol.real
            if sol>0:
                val = np.polyval(poly,sol)
                if val<current_best:
                    current_best = val
                    best_x = sol
    if current_best==np.Inf:
        print("No solution to scaling !!!")
        return tensor, None

    # We have the optimal scale
    for i in range(n_modes):
        factors[i] *= best_x

    if format_tensor=="tucker":
        return tl.tucker_tensor.TuckerTensor((factors[-1], factors[:-1])), best_x
    return tl.cp_tensor.CPTensor((None, factors)), best_x
