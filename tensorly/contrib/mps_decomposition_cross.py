import tensorly as tl
from ..mps_tensor import mps_to_tensor
from numpy import asarray

import numpy as np
import numpy.random as npr
from scipy import linalg as scla

npr.seed(1)

def matrix_product_state_cross(input_tensor, rank, delta=1e-5, max_iter=100, mv_eps=1e-5, mv_maxit=100):
    """MPS (tensor-train) decomposition via cross-approximation [1]

    Acknowledgement: the main body of the code is modified based on TensorToolbox by Daniele Bigoni
    ----------------------------------------------------------------------
    Pseudo-code [2]:
    1. Intialization d cores and column indices
    2. while (error > delta)
    3.    update the tensor-train from left to right:
                for Core 1 to Core d
                    approximate the skeleton-decomposition by QR and maxvol
    4.    update the tensor-train from right to left:
                for Core d to Core 1
                    approximate the skeleton-decomposition by QR and maxvol
    5. end while

    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : {int, int list}
            maximum allowable MPS rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor

    :param list rank: list of upper ranks
    :param list col_idx: column indices for skeleton-decomposition
    :param list row_idx: row indices for skeleton-decomposition
    :param list col_idx: list (d-1) of lists of init indices
    :param float mv_eps: MaxVol accuracy
    :param int mv_maxit: maximum number of iterations for MaxVol

    Returns
    -------
    factors : MPS factors
              order-3 tensors of the MPS decomposition

    References
    ----------
    .. [1] Ivan Oseledets and Eugene Tyrtyshnikov.  Tt-cross approximation for multidimensional arrays.
            LinearAlgebra and its Applications, 432(1):70–88, 2010.
    .. [2] Sergey Dolgov and Robert Scheichl. A hybrid alternating least squares–tt cross algorithm for parametricpdes.
            arXiv preprint arXiv:1707.04562, 2017.
    """

    # unfortunately we are not able to work on tensorflow yet
    # if tl.get_backend()=='tensorflow':
    #     tl.set_backend('numpy')
    #     input_tensor = tl.to_numpy(input_tensor)

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

    # Initialization
    if rank[0] != 1:
        print('Provided rank[0] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[0] to 1.'.format(rank[0]))
        rank[0] = 1
    if rank[-1] != 1:
        print('Provided rank[-1] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[-1] to 1.'.format(rank[0]))


    # random selection of column indices
    col_idx = [None] * d
    for k_col_idx in range(d-1):
        col_idx[k_col_idx] = []
        for i in range(rank[k_col_idx+1]):
            newidx = tuple( [ npr.choice(range(n[j])) for j in range(k_col_idx+1,d) ] )
            while newidx in col_idx[k_col_idx]:
                newidx = tuple( [ npr.choice(range(n[j])) for j in range(k_col_idx+1,d) ] )

            col_idx[k_col_idx].append(newidx)

    # Start the while-loop
    factor_old = [ tl.zeros((rank[k],n[k],rank[k+1])) for k in range(d) ]
    factor_new = [ tl.tensor(npr.random((rank[k],n[k],rank[k+1]))) for k in range(d) ]

    iter = 0
    while iter < max_iter and tl.norm(mps_to_tensor(factor_old)-mps_to_tensor(factor_new), 2) > delta * tl.norm(mps_to_tensor(factor_new), 2):
        iter += 1
        factor_old = factor_new
        factor_new = [None for i in range(d)]

        ######################################
        # left-to-right step
        LeftToRight_fiberlist = []
        # list row_idx: list (d-1) of lists of left indices
        row_idx = [[()]]
        for k in range(d-1):
            (IT, fibers_list, Q, QsqInv) = left_right_ttcross_step(input_tensor, iter, k, rank, row_idx, col_idx,  mv_eps, mv_maxit)
            LeftToRight_fiberlist.extend( fibers_list )
            row_idx.append(IT)

        # end left-to-right step
        ###############################################

        ###############################################
        # right-to-left step
        RightToLeft_fiberlist = []
        # list col_idx: list (d-1) of lists of right indices
        col_idx = [None] * d
        col_idx[-1] = [()]
        for k in range(d,1,-1):
            (JT, fibers_list, Q, QsqInv) = right_left_ttcross_step(input_tensor, iter, k,rank, row_idx, col_idx, mv_eps, mv_maxit)
            RightToLeft_fiberlist.extend( fibers_list )
            col_idx[k-2] = JT

            # Compute cores
            try:
                factor_new[k-1] = tl.transpose(tl.dot(Q,QsqInv))
                factor_new[k-1] = tl.reshape(factor_new[k-1] ,(rank[k-1], n[k-1], rank[k]) )
            except:
                raise(ValueError("The rank is too large compared to the size of the tensor. Try with small rank."))

        # Add the last core
        idx = (slice(None,None,None),) + tuple(zip(*col_idx[0]))

        C = input_tensor[ idx]
        C = tl.reshape(C,(n[0], 1, rank[1]))
        C = tl.transpose(C, (1,0,2) )

        factor_new[0] = C

        # end right-to-left step
        ################################################

    # check convergence
    if iter >= max_iter:
        raise ValueError('Maximum number of iterations reached.')
    if tl.norm(mps_to_tensor(factor_old)-mps_to_tensor(factor_new),2) > delta * tl.norm(mps_to_tensor(factor_new), 2):
        raise ValueError('Low Rank Approximation algorithm did not converge.')

    return factor_new

def left_right_ttcross_step(input_tensor, iter, k, rs, row_idx, col_idx, mv_eps, mv_maxit):
    """ Compute one step of left-right sweep of ttcross.

    :param int iter: the actual ttcross iteration
    :param int k: the actual sweep iteration
    :param list rs: list of upper ranks (d)
    :param list row_idx: list (d-1) of lists of left indices
    :param list col_idx: list (d-1) of lists of right indices
    :param float mv_eps: MaxVol accuracy
    :param int mv_maxit: maximum number of iterations for MaxVol

    :returns: tuple containing: ``(IT,fibers_list,Q,QsqInv)``, the list of new indices, the used fibers, the Q matrix and the inverse of the maxvol submatrix
    """

    n = tl.shape(input_tensor)
    d = tl.ndim(input_tensor)
    fibers_list = []

    # Extract fibers
    for i in range(rs[k]):
        for j in range(rs[k+1]):
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


    C = input_tensor[ idx]
    if k == 0:
        C = tl.reshape(C, (n[k], rs[k], rs[k+1]))
        C = tl.transpose(C, (1,0,2) )
    else:
        C = tl.reshape(C, (rs[k], rs[k+1], n[k]))
        C = tl.transpose(C, (0,2,1) )

    C = tl.reshape(C, (rs[k] * n[k], rs[k+1] ))

    # Compute QR decomposition
    (Q,R) = tl.qr(C)

    # Maxvol
    (I,QsqInv) = maxvol(Q,mv_eps,mv_maxit)
    QsqInv = tl.tensor(QsqInv)

    # Retrive indices in folded tensor
    IC = [ idxfold( [rs[k],n[k]], idx ) for idx in I ] # First retrive idx in folded C
    IT = [row_idx[k][ic[0]] + (ic[1],) for ic in IC] # Then reconstruct the idx in the tensor

    return (IT, fibers_list, Q, QsqInv)

def right_left_ttcross_step(input_tensor, iter, k, rank, row_idx, col_idx, mv_eps, mv_maxit):
    """ Compute one step of right-left sweep of ttcross.

    :param int iter: the actual ttcross iteration
    :param int k: the actual sweep iteration
    :param list rank: list of upper ranks (d)
    :param list row_idx: list (d-1) of lists of left indices
    :param list col_idx: list (d-1) of lists of right indices
    :param float mv_eps: MaxVol accuracy
    :param int mv_maxit: maximum number of iterations for MaxVol

    :returns: tuple containing: ``(JT,fibers_list,Q,QsqInv)``, the list of new indices, the used fibers, the Q matrix and the inverse of the maxvol submatrix
    """
    n = tl.shape(input_tensor)
    d = tl.ndim(input_tensor)
    fibers_list = []

    # Extract fibers
    for i in range(rank[k-1]):
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

    C = tl.reshape(C, (rank[k - 1], rank[k], n[k - 1]))
    C = tl.transpose(C, (0,2,1) )

    C = tl.reshape(C, (rank[k - 1], n[k - 1] * rank[k]))
    C = tl.transpose(C)

    # Compute QR decomposition
    (Q,R) = tl.qr(C)
    # Maxvol
    (J,QsqInv) = maxvol(Q,mv_eps,mv_maxit)
    QsqInv = tl.tensor(QsqInv)

    # Retrive indices in folded tensor
    JC = [idxfold([n[k-1], rank[k]], idx) for idx in J] # First retrive idx in folded C
    JT = [(jc[0],) + col_idx[k - 1][jc[1]] for jc in JC] # Then reconstruct the idx in the tensor

    return (JT, fibers_list, Q, QsqInv)

def idxfold(dlist,idx):
    """ Find the index corresponding to the folded version of a tensor from the flatten version

    :param list,int dlist: list of integers containing the dimensions of the tensor
    :param int idx: tensor flatten index

    :returns: list of int -- the index for the folded version
    """
    n = len(dlist)

    cc = [1]
    for val in reversed(dlist): cc.append( cc[-1] * val )
    if idx >= cc[-1]: raise ValueError("Index out of bounds")

    ii = []
    tmp = idx

    for i in range(n):
        ii.append( tmp//cc[n-i-1] )
        tmp = tmp % cc[n-i-1]

    return tuple(ii)

def maxvol(A,delta=1e-2,maxit=100):
    """ Find the rxr submatrix of maximal volume in A(nxr), n>=r

    :param ndarray A: two dimensional array with (n,r)=shape(A) where r<=n
    :param float delta: stopping criterion [default=1e-2]
    :param int maxit: maximum number of iterations [default=100]

    :returns: ``(I,A_sq_inv,it)`` where ``I`` is the list or rows of A forming the matrix with maximal volume,
    ``A_sq_inv`` is the inverse of the matrix with maximal volume and
    ``it`` is the number of iterations to convergence
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
    # P = tl.tensor(P); L = tl.tensor(L); U = tl.tensor(U)
    # Reorder A so that the non-singular matrix is on top

    I = tl.arange(n) # set of swapping indices
    I = tl.dot(tl.tensor(P.T),I)
    I = tl.int(I)
    if tl.get_backend() =='mxnet':
        I = tl.to_numpy(I)
        A = tl.tensor(A)

    # Compute inverse of Asq: Asq^-1 = (PLU)^-1
    LU = L[:r,:r] - np.eye(r) + U
    A_sq_inv = scla.lu_solve((LU,np.arange(r)), np.eye(r))
    A_sq_inv = tl.tensor(A_sq_inv)
    # Compute B
    B = tl.dot(A[I,:], A_sq_inv)
    B = tl.tensor(B)
    if tl.get_backend() =='mxnet':
        A_sq_inv = tl.tensor(A_sq_inv)
        I = tl.tensor(I); I = tl.int(I)

    maxrow = tl.argmax(tl.abs(B),axis=0)[0]
    maxcol = tl.argmax(tl.abs(B),axis=1)[0]
    maxB = tl.abs(B)[maxrow,maxcol]

    it = 0
    eps = 1.+ delta
    while it < maxit and maxB > eps:
        it += 1

        # Update A_sq_inv
        q_tensor = tl.zeros((r,1))

        vT = A[I[maxrow],:] - A[I[maxcol],:]
        if len(vT.shape) == 1:
            vT = tl.reshape(vT,(1, vT.shape[0]))

        # Eq (8) in "How to find a good submatrix"
        A_sq_inv -= tl.dot(tl.dot(A_sq_inv,q_tensor), tl.dot(vT,A_sq_inv)) / (1. + tl.dot(vT,tl.dot(A_sq_inv,q_tensor)))

        # Update B using Sherman-Woodbury-Morrison formula
        Bj = B[:,maxcol]
        Bi = B[maxrow,:]
        if len(Bi.shape) == 1:
            Bj = tl.reshape(Bj, (Bj.shape[0],1))
            Bi = tl.reshape(Bi, (1,Bi.shape[0]))
        Bj[maxrow, 0] = Bj[maxrow, 0] + 1.
        Bi[0,maxcol] -= 1.
        B[r:,:] -= tl.dot(Bj[r:],Bi)/B[maxrow,maxcol]

        # Update index of maxvol matrix I
        tmp = I[maxcol]
        I[maxcol] = I[maxrow]
        I[maxrow] = tmp

        # Find new maximum in B
        maxrow = tl.argmax(tl.abs(B),axis=0)[0]
        maxcol = tl.argmax(tl.abs(B),axis=1)[0]
        maxB = tl.abs(B)[maxrow,maxcol]

    if maxB > eps:
        raise ValueError('Maxvol algorithm did not converge.')

    # Return max-vol submatrix Asq
    I = tl.to_numpy(I)
    return (list(I[:r]), A_sq_inv)
