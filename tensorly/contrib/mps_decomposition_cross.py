import tensorly as tl
from ..mps_tensor import mps_to_tensor

import numpy as np
import numpy.random as npr
from scipy import linalg as scla

# npr.seed(1)

def matrix_product_state_cross(input_tensor, rank, delta=1e-5, max_iter=100, mv_eps=1e-5, mv_max_iter=100):
    """MPS (tensor-train) decomposition via cross-approximation [1]

        Decomposes `input_tensor` into a sequence of order-3 tensors of given rank. (factors)
        Advantage: faster
            The main advantage of TTcross is that it doesn't need to evaluate all the entries of the tensor.
            For a n^d tensor, SVD needs O(n^d) runtime, but TTcross' runtime is linear in n and d, which makes it feasible in high dimension.
        Disadvantage: less accurate
            TTcross may underestimate the error, since it only evaluates partial entries of the tensor.
            Besides, in contrast to its practical fast performance, there is no theoretical guarantee of it convergence.

    Parameters
    ----------
    input_tensor : tensorly.tensor
            The tensor to decompose.
    rank : {int, int list}
            maximum allowable MPS rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    delta : float
            accuracy threshold for outer while-loop
    max_iter : int
            maximum iterations of outer while-loop (the 'crosses' or 'sweeps' sampled)
    mv_eps: float
            accuracy threshold for max-volumn algorithm
    mv_max_iter: int
            maximum iterations of  max-volumn algorithm

    Returns
    -------
    factors : MPS factors
              order-3 tensors of the MPS decomposition

    Examples
    --------

    Generate a 5^3 tensor, and decompose it into tensor-train of 3 factors, with rank = [1,3,3,1]
    >>> tensor = tl.tensor(np.arange(5**3).reshape(5,5,5))
    >>> rank = [1, 3, 3, 1]
    >>> factors = matrix_product_state_cross(tensor, rank)
    print the first core:
    >>> print(factors[0])
    .[[[ 24.   0.   4.]
      [ 49.  25.  29.]
      [ 74.  50.  54.]
      [ 99.  75.  79.]
      [124. 100. 104.]]]

    Notes
    -----
    Pseudo-code [2]:
    1. Initialization d cores and column indices
    2. while (error > delta)
    3.    update the tensor-train from left to right:
                for Core 1 to Core d
                    approximate the skeleton-decomposition by QR and maxvol
    4.    update the tensor-train from right to left:
                for Core d to Core 1
                    approximate the skeleton-decomposition by QR and maxvol
    5. end while

    Acknowledgement: the main body of the code is modified based on TensorToolbox by Daniele Bigoni

    References
    ----------
    .. [1] Ivan Oseledets and Eugene Tyrtyshnikov.  Tt-cross approximation for multidimensional arrays.
            LinearAlgebra and its Applications, 432(1):70–88, 2010.
    .. [2] Sergey Dolgov and Robert Scheichl. A hybrid alternating least squares–tt cross algorithm for parametricpdes.
            arXiv preprint arXiv:1707.04562, 2017.
    """

    # Check user input for errors
    n = tl.shape(input_tensor)
    d = tl.ndim(input_tensor)

    if isinstance(rank, int):
        rank = [rank] * (d+1)
    elif d+1 != len(rank):
        message = 'Provided incorrect number of ranks. Should verify len(rank) == tl.ndim(tensor)+1, but len(rank) = {} while tl.ndim(tensor) + 1  = {}'.format(
            len(rank), d)
        raise(ValueError(message))

    # Make sure iter's not a tuple but a list
    rank = list(rank)

    # Initialize rank
    if rank[0] != 1:
        print('Provided rank[0] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[0] to 1.'.format(rank[0]))
        rank[0] = 1
    if rank[-1] != 1:
        print('Provided rank[-1] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[-1] to 1.'.format(rank[0]))


    #list col_idx: column indices (right indices) for skeleton-decomposition: indicate which columns used in each core.
    #list row_idx: row indices    (left indices)  for skeleton-decomposition: indicate which rows used in each core.

    # Initialize indice: random selection of column indices
    col_idx = [None] * d
    for k_col_idx in range(d-1):
        col_idx[k_col_idx] = []
        for i in range(rank[k_col_idx+1]):
            newidx = tuple( [ npr.choice(range(n[j])) for j in range(k_col_idx+1,d) ] )
            while newidx in col_idx[k_col_idx]:
                newidx = tuple( [ npr.choice(range(n[j])) for j in range(k_col_idx+1,d) ] )

            col_idx[k_col_idx].append(newidx)

    # Initialize the cores of tensor-train
    factor_old = [ tl.zeros((rank[k],n[k],rank[k+1])) for k in range(d) ]
    factor_new = [ tl.tensor(npr.random((rank[k],n[k],rank[k+1]))) for k in range(d) ]

    iter = 0

    error = tl.norm(mps_to_tensor(factor_old)-mps_to_tensor(factor_new), 2)
    threshold = delta * tl.norm(mps_to_tensor(factor_new), 2)
    for iter in range(max_iter):
        if error < threshold:
            break

        factor_old = factor_new
        factor_new = [None for i in range(d)]

        ######################################
        # left-to-right step
        LeftToRight_fiberlist = []
        # list row_idx: list (d-1) of lists of left indices
        row_idx = [[()]]
        for k in range(d-1):
            (next_row_idx, fibers_list, Q, Q_inv) = left_right_ttcross_step(input_tensor, k, rank, row_idx, col_idx, mv_eps, mv_max_iter)
            # update row indices
            LeftToRight_fiberlist.extend( fibers_list )
            row_idx.append(next_row_idx)

        # end left-to-right step
        ###############################################

        ###############################################
        # right-to-left step
        RightToLeft_fiberlist = []
        # list col_idx: list (d-1) of lists of right indices
        col_idx = [None] * d
        col_idx[-1] = [()]
        for k in range(d,1,-1):
            (next_col_idx, fibers_list, Q, Q_inv) = right_left_ttcross_step(input_tensor, k, rank, row_idx, col_idx, mv_eps, mv_max_iter)
            # update col indices
            RightToLeft_fiberlist.extend( fibers_list )
            col_idx[k-2] = next_col_idx

            # Compute cores
            try:
                factor_new[k-1] = tl.transpose(tl.dot(Q,Q_inv))
                factor_new[k-1] = tl.reshape(factor_new[k-1] ,(rank[k-1], n[k-1], rank[k]) )
            except:
                # The rank should not be larger than the input tensor's size
                raise(ValueError("The rank is too large compared to the size of the tensor. Try with small rank."))

        # Add the last core
        idx = (slice(None,None,None),) + tuple(zip(*col_idx[0]))

        C = input_tensor[ idx]
        C = tl.reshape(C,(n[0], 1, rank[1]))
        C = tl.transpose(C, (1,0,2) )

        factor_new[0] = C

        # end right-to-left step
        ################################################

        # check the error for while-loop
        error = tl.norm(mps_to_tensor(factor_old) - mps_to_tensor(factor_new), 2)
        threshold = delta * tl.norm(mps_to_tensor(factor_new), 2)

    # check convergence
    if iter >= max_iter:
        raise ValueError('Maximum number of iterations reached.')
    if tl.norm(mps_to_tensor(factor_old)-mps_to_tensor(factor_new),2) > delta * tl.norm(mps_to_tensor(factor_new), 2):
        raise ValueError('Low Rank Approximation algorithm did not converge.')

    return factor_new

def left_right_ttcross_step(input_tensor, k, rank, row_idx, col_idx, mv_eps, mv_max_iter):
    """ Compute the next (right) core's row indices by QR decomposition.
    
            For the current Tensor train core, we use the row indices and col indices to extract the entries from the input tensor
            and compute the next core's row indices by QR and max volume algorithm.

    Parameters
    ----------

    k: int
            the actual sweep iteration
    rank: list of int
            list of upper ranks (d)
    row_idx: list of list of int
            list (d-1) of lists of left indices
    col_idx: list of list of int
            list (d-1) of lists of right indices
    mv_eps: float
            MaxVol accuracy
    mv_max_iter: int
            maximum number of iterations for MaxVol

    Returns
    -------
    next_row_idx : list of int
            the list of new row indices,
    fibers_list : list of slice
            the used fibers,
    Q : matrix
            the Q matrix from QR decomposition
    Q_inv : matrix
        the inverse of the maxvol submatrix
    """

    n = tl.shape(input_tensor)
    d = tl.ndim(input_tensor)
    fibers_list = []

    # Extract fibers according to the row and col indices
    for i in range(rank[k]):
        for j in range(rank[k + 1]):
            fiber = row_idx[k][i] + (slice(None, None, None),) + col_idx[k][j]
            fibers_list.append(fiber)
    if k == 0:      # Is[k] will be empty
        idx = (slice(None,None,None),) + tuple(zip(*col_idx[k]))
    else:
        idx = [ [] for i in range(d) ]
        for lidx in row_idx[k]:
            for ridx in col_idx[k]:
                for j,jj in enumerate(lidx): idx[j].append(jj)
                for j,jj in enumerate(ridx): idx[len(lidx)+1+j].append(jj)
        idx[k] = slice(None,None,None)
        idx = tuple(idx)

    # Extract the core
    C = input_tensor[ idx]
    # shape the core as a 3-d cube
    if k == 0:
        C = tl.reshape(C, (n[k], rank[k], rank[k + 1]))
        C = tl.transpose(C, (1,0,2) )
    else:
        C = tl.reshape(C, (rank[k], rank[k + 1], n[k]))
        C = tl.transpose(C, (0,2,1) )

    # merge r_k and n_k, get a matrix
    C = tl.reshape(C, (rank[k] * n[k], rank[k + 1]))

    # Compute QR decomposition
    (Q,R) = tl.qr(C)

    # Maxvol
    (I, Q_inv) = maxvol(Q, mv_eps, mv_max_iter)
    Q_inv = tl.tensor(Q_inv)

    # Retrive indices in folded tensor
    new_idx = [idxfold([rank[k], n[k]], idx) for idx in I] # First retrive idx in folded C
    next_row_idx = [row_idx[k][ic[0]] + (ic[1],) for ic in new_idx] # Then reconstruct the idx in the tensor

    return (next_row_idx, fibers_list, Q, Q_inv)

def right_left_ttcross_step(input_tensor, k, rank, row_idx, col_idx, mv_eps, mv_max_iter):
    """ Compute the next (left) core's col indices by QR decomposition.

            For the current Tensor train core, we use the row indices and col indices to extract the entries from the input tensor
            and compute the next core's col indices by QR and max volume algorithm.

    Parameters
    ----------

    k: int
            the actual sweep iteration
    rank: list of int
            list of upper rank (d)
    row_idx: list of list of int
            list (d-1) of lists of left indices
    col_idx: list of list of int
            list (d-1) of lists of right indices
    mv_eps: float
            MaxVol accuracy
    mv_max_iter: int
            maximum number of iterations for MaxVol

    Returns
    -------
    next_col_idx : list of int
            the list of new col indices,
    fibers_list : list of slice
            the used fibers,
    Q : matrix
            the Q matrix from QR decomposition
    Q_inv : matrix
        the inverse of the maxvol submatrix
    """

    n = tl.shape(input_tensor)
    d = tl.ndim(input_tensor)
    fibers_list = []

    # Extract fibers
    for i in range(rank[k - 1]):
        for j in range(rank[k]):
            fiber = row_idx[k - 1][i] + (slice(None, None, None),) + col_idx[k - 1][j]
            fibers_list.append(fiber)

    if k == d:      # Is[k] will be empty
        idx = tuple(zip(*row_idx[k - 1])) + (slice(None, None, None),)
    else:
        idx = [ [] for i in range(d) ]
        for lidx in row_idx[k-1]:
            for ridx in col_idx[k-1]:
                for j,jj in enumerate(lidx): idx[j].append(jj)
                for j,jj in enumerate(ridx): idx[len(lidx)+1+j].append(jj)
        idx[k-1] = slice(None,None,None)
        idx = tuple(idx)

    C = input_tensor[ idx]
    # shape the core as a 3-d cube
    C = tl.reshape(C, (rank[k - 1], rank[k], n[k - 1]))
    C = tl.transpose(C, (0,2,1) )
    # merge n_{k-1} and r_k, get a matrix
    C = tl.reshape(C, (rank[k - 1], n[k - 1] * rank[k]))
    C = tl.transpose(C)

    # Compute QR decomposition
    (Q,R) = tl.qr(C)
    # Maxvol
    (J,Q_inv) = maxvol(Q, mv_eps, mv_max_iter)
    Q_inv = tl.tensor(Q_inv)

    # Retrive indices in folded tensor
    new_idx = [idxfold([n[k-1], rank[k]], idx) for idx in J] # First retrive idx in folded C
    next_col_idx = [(jc[0],) + col_idx[k - 1][jc[1]] for jc in new_idx] # Then reconstruct the idx in the tensor

    return (next_col_idx, fibers_list, Q, Q_inv)

def idxfold(dlist,idx):
    """ Find the index corresponding to the folded version of a tensor from the flatten version

    Parameters
    ----------
    dlist: list of int
            list of integers containing the dimensions of the tensor
    idx: int
            tensor flatten index

    Returns
    -------
    idx_folded: list of int -- the index for the folded version

    """
    n = len(dlist)

    cc = [1]
    for val in reversed(dlist):
        cc.append( cc[-1] * val )
    if idx >= cc[-1]: raise ValueError("Index out of bounds")

    idx_folded = []
    tmp = idx

    for i in range(n):
        idx_folded.append( tmp//cc[n-i-1] )
        tmp = tmp % cc[n-i-1]

    return tuple(idx_folded)

def maxvol(A, delta=1e-2, max_iter=100):
    """ Find the rxr submatrix of maximal volume in A(nxr), n>=r

    Parameters
    ----------

    A: matrix
            The matrix to find maximal volume
    delta: delta
            stopping criterion
    max_iter: int
            maximum number of iterations

    Returns
    -------
    row_idx: list of int
            is the list or rows of A forming the matrix with maximal volume,
    A_inv: matrix
            is the inverse of the matrix with maximal volume.

    References
    ----------
    S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov, N. L. Zamarashkin.
    How to find a good submatrix.Goreinov, S. A., et al.
    Matrix Methods: Theory, Algorithms and Applications: Dedicated to the Memory of Gene Golub. 2010. 247-256.
    """

    (n,r) = tl.shape(A)

    if r>n :
        raise TypeError("maxvol: A(nxr) must be a thin matrix, i.e. n>=r")

    # Find an arbitrary non-singular rxr matrix in A
    if tl.get_backend() =='mxnet':
        A = tl.to_numpy(A)

    (P,L,U) = scla.lu(A)
    # Check singularity
    if tl.min(tl.abs(tl.tensor(np.diag(U)))) < np.spacing(1):
        raise ValueError("maxvol: Matrix A is singular")
    # Reorder A so that the non-singular matrix is on top

    row_idx = tl.arange(n) # set of swapping indices
    row_idx = tl.dot(tl.tensor(P.T),row_idx)
    row_idx = tl.int(row_idx)
    if tl.get_backend() =='mxnet':
        row_idx = tl.to_numpy(row_idx)
        A = tl.tensor(A)

    # Compute inverse of Asq: Asq^-1 = (PLU)^-1
    LU = L[:r,:r] - np.eye(r) + U
    A_inv = scla.lu_solve((LU,np.arange(r)), np.eye(r))
    A_inv = tl.tensor(A_inv)
    # Compute B
    B = tl.dot(A[row_idx,:], A_inv)
    B = tl.tensor(B)
    if tl.get_backend() =='mxnet':
        A_inv = tl.tensor(A_inv)
        row_idx = tl.tensor(row_idx); row_idx = tl.int(row_idx)

    maxrow = tl.argmax(tl.abs(B),axis=0)[0]
    maxcol = tl.argmax(tl.abs(B),axis=1)[0]
    maxB = tl.abs(B)[maxrow,maxcol]

    it = 0
    eps = 1.+ delta
    for iter in range(max_iter):
        if maxB < eps:
            break

        # Update A_inv
        q_tensor = tl.zeros((r,1))

        vT = A[row_idx[maxrow],:] - A[row_idx[maxcol],:]
        if len(vT.shape) == 1:
            vT = tl.reshape(vT,(1, vT.shape[0]))

        # Eq (8) in "How to find a good submatrix"
        A_inv -= tl.dot(tl.dot(A_inv,q_tensor), tl.dot(vT,A_inv)) / (1. + tl.dot(vT,tl.dot(A_inv,q_tensor)))

        # Update B using Sherman-Woodbury-Morrison formula
        Bj = B[:,maxcol]
        Bi = B[maxrow,:]
        if len(Bi.shape) == 1:
            Bj = tl.reshape(Bj, (Bj.shape[0],1))
            Bi = tl.reshape(Bi, (1,Bi.shape[0]))
        Bj[maxrow, 0] = Bj[maxrow, 0] + 1.
        Bi[0,maxcol] -= 1.
        B[r:,:] -= tl.dot(Bj[r:],Bi)/B[maxrow,maxcol]

        # Update index of maxvol matrix row_idx
        tmp = row_idx[maxcol]
        row_idx[maxcol] = row_idx[maxrow]
        row_idx[maxrow] = tmp

        # Find new maximum in B
        maxrow = tl.argmax(tl.abs(B),axis=0)[0]
        maxcol = tl.argmax(tl.abs(B),axis=1)[0]
        maxB = tl.abs(B)[maxrow,maxcol]

    if maxB > eps:
        raise ValueError('Maxvol algorithm did not converge.')

    # Return max-vol submatrix Asq
    row_idx = tl.to_numpy(row_idx)
    return list(row_idx[:r]), A_inv
