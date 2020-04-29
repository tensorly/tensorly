import numpy as np
import tensorflow as tf
from . import Backend, KruskalTensor

def is_sparse(x):
    return isinstance(x, tf.sparse.SparseTensor)

class TensorflowBackend(Backend):
    backend_name = 'tensorflow.sparse'

    @staticmethod
    def tensor(data, dtype=np.float32, device=None, device_id=None):
        if isinstance(data, tf.sparse.SparseTensor):
            return data

# This function is easy to decorate with Numba '@njit(parallel=True)' directive for ~10x improvement in performance
def mttkrp(self, values, indices, factors, n, rank, dims):
    output = np.zeros((dims[n], rank))
    # if decorating with Numba '@njit(parallel=True)' directive, replcase range with numba.prange
    for l in range(len(values)):
        cur_index = indices[l]
        prod = [values[l]] * rank  # makes the value into a row

        for mode, cv in enumerate(cur_index):  # does elementwise row multiplications
            if (mode != n):
                for r in range(rank):
                    prod[r] *= factors[mode][cv][r]

        for r in range(rank):
            output[cur_index[n]][r] += prod[r]

    return output

def parafac(tensor, rank, n_iter_max=100, verbose=False):
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
        Computes a rank-`rank` decomposition of `tensor` [1]_ such that,

            ``tensor = [|weights; factors[0], ..., factors[-1] |]``.

        Parameters
        ----------
        tensor : tf.sparse.SparseTensor
        rank  : int
            Number of components.
        n_iter_max : int
            Maximum number of iteration
        verbose: bool
            Wether or not to print progress at every iteration

        Returns
        -------
        KruskalTensor : (weight, factors)
            * weights : 1D array of shape (rank, )
                all ones if normalize_factors is False (default),
                weights of the (normalized) factors otherwise
            * factors : List of factors of the CP decomposition element `i` is of shape
                (tensor.shape[i], rank)

        References
        ----------
        .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
           SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
        """

    dims = tensor.shape.as_list()
    nd = len(dims)
    factors = [np.random.random((d, rank)) for d in dims]
    weights = np.ones((1, rank))

    for iteration in range(n_iter_max):
        if verbose:
            print("finished {} iterations".format(iteration+1), end="\r")
        for n in range(nd):

            # the following block calculates inverse of the hadamard product
            h = np.matmul(weights.T, weights)
            for i, f in enumerate(factors):
                if i != n:
                    h *= np.matmul(f.T, f)
            vinv = np.linalg.pinv(h)

            # the following block calculates An by doing MTTKRP and multiplying it by the inverse of the hadamard
            vals = tensor.values.numpy()
            idxs = tensor.indices.numpy()
            mk = mttkrp(vals, idxs, factors, n, rank, dims)

            wmk = np.multiply(mk, weights[0])  #multiply each column by the weights
            An = np.matmul(wmk, vinv)          #nth factor

            # the following block normalizes the columns and stored
            weight = np.linalg.norm(An, axis=0)
            b = np.where(weight < 1e-12, 1, weight)
            weights[0] *= b #avoids dividing by small weights, reduces error
            An /= b

            factors[n] = An

    return KruskalTensor(weights, factors)