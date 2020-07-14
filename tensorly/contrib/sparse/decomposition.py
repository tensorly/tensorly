from ...decomposition import (tucker, partial_tucker, non_negative_tucker,
                              non_negative_parafac, robust_pca)
from ...decomposition import parafac as non_sparse_parafac
from .core import wrap
import numpy as np
from .kruskal_tensor import sparse_mttkrp as mttkrp
from ...kruskal_tensor import KruskalTensor

tucker = wrap(tucker)
partial_tucker = wrap(partial_tucker)
non_negative_tucker = wrap(non_negative_tucker)
robust_pca = wrap(robust_pca)
non_negative_parafac = wrap(non_negative_parafac)

def parafac(tensor, rank, n_iter_max=100, init='svd', svd='numpy_svd',\
            normalize_factors=False, orthogonalise=False,\
            tol=1e-8, random_state=None,\
            verbose=0, return_errors=False,\
            non_negative=False,\
            sparsity = None,\
            l2_reg = 0,  mask=None,\
            cvg_criterion = 'abs_rec_error'):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
       Computes a rank-`rank` decomposition of `tensor` [1]_ such that,

           ``tensor = [|weights; factors[0], ..., factors[-1] |]``.

       Parameters
       ----------
       tensor : ndarray
       rank  : int
           Number of components.
       n_iter_max : int
           Maximum number of iteration
       init : {'svd', 'random'}, optional
           Type of factor matrix initialization. See `initialize_factors`.
       svd : str, default is 'numpy_svd'
           function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
       normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
           of shape (rank, ), which will contain the norms of the factors
       tol : float, optional
           (Default: 1e-6) Relative reconstruction error tolerance. The
           algorithm is considered to have found the global minimum when the
           reconstruction error is less than `tol`.
       random_state : {None, int, np.random.RandomState}
       verbose : int, optional
           Level of verbosity
       return_errors : bool, optional
           Activate return of iteration errors
       mask : ndarray
           array of booleans with the same shape as ``tensor`` should be 0 where
           the values are missing and 1 everywhere else. Note:  if tensor is
           sparse, then mask should also be sparse with a fill value of 1 (or
           True). Allows for missing values [2]_
       cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
          Stopping criterion for ALS, works if `tol` is not None.
          If 'rec_error',  ALS stops at current iteration if (previous rec_error - current rec_error) < tol.
          If 'abs_rec_error', ALS terminates when |previous rec_error - current rec_error| < tol.
       sparsity : float or int
           If `sparsity` is not None, we approximate tensor as a sum of low_rank_component and sparse_component, where low_rank_component = kruskal_to_tensor((weights, factors)). `sparsity` denotes desired fraction or number of non-zero elements in the sparse_component of the `tensor`.

       Returns
       -------
       KruskalTensor : (weight, factors)
           * weights : 1D array of shape (rank, )
               all ones if normalize_factors is False (default),
               weights of the (normalized) factors otherwise
           * factors : List of factors of the CP decomposition element `i` is of shape
               (tensor.shape[i], rank)
           * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.

       errors : list
           A list of reconstruction errors at each iteration of the algorithms.

       References
       ----------
       .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
          SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.

       .. [2] Tomasi, Giorgio, and Rasmus Bro. "PARAFAC and missing values."
               Chemometrics and Intelligent Laboratory Systems 75.2 (2005): 163-180.

       """
    try:
        dims = tensor.shape.as_list()
        nd = len(dims)
        factors = [np.random.random((d, rank)) for d in dims]
        weights = np.ones((1, rank))

        for iteration in range(n_iter_max):
            if verbose:
                print("finished {} iterations".format(iteration + 1), end="\r")
            for n in range(nd):

                # the following block calculates inverse of the hadamard product
                h = np.matmul(weights.T, weights)
                for i, f in enumerate(factors):
                    if i != n:
                        h *= np.matmul(f.T, f)
                vinv = np.linalg.pinv(h)

                # the following block calculates An by doing MTTKRP and multiplying it by the inverse of the hadamard
                mk = mttkrp(tensor, factors, n, rank, dims)

                wmk = np.multiply(mk, weights[0])  # multiply each column by the weights
                An = np.matmul(wmk, vinv)  # nth factor

                # the following block normalizes the columns and stored
                weight = np.linalg.norm(An, axis=0)
                b = np.where(weight < 1e-12, 1, weight)
                weights[0] *= b  # avoids dividing by small weights, reduces error
                An /= b

                factors[n] = An

        return KruskalTensor((weights[0], factors))
    except NotImplementedError:
        print("non sparse input")
        return non_sparse_parafac(tensor, rank, n_iter_max=n_iter_max,
                                  init=init, svd=svd, normalize_factors=normalize_factors,
                                  orthogonalise=orthogonalise, tol=tol, random_state=random_state,
                                  verbose=verbose, return_errors=return_errors,
                                  non_negative=non_negative, sparsity=sparsity,
                                  l2_reg=l2_reg, mask=mask, cvg_criterion=cvg_criterion)
