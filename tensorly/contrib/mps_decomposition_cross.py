import tensorly as tl
from ..mps_tensor import mps_to_tensor
from numpy import asarray

import numpy as np
import numpy.random as npr
from scipy import linalg as scla



def matrix_product_state_cross(input_tensor, rank, delta=1e-5, maxit=100, mv_eps=1e-5, mv_maxit=100):
    """MPS (tensor-train) decomposition via cross-approximation [1]

    Acknowledgement: the main body of the code is modified based on TensorToolbox by Daniele Bigoni

    Pseudo-code [2]:
    1. Intialization
    2. while (error > delta)
    3.    update the tensor-train from left to right by QR and maxvol
    4.    update the tensor-train from right to left by QR and maxvol

    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : {int, int list}
            maximum allowable MPS rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor

    :param list rank: list of upper ranks
    :param list Jinit: list (d-1) of lists of init indices
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

    # Check user input for errors
    n = input_tensor.shape
    d = len(n)

    if isinstance(rank, int):
        rank = [rank] * (d+1)
    elif d+1 != len(rank):
        message = 'Provided incorrect number of ranks. Should verify len(rank) == tl.ndim(tensor)+1, but len(rank) = {} while tl.ndim(tensor) + 1  = {}'.format(
            len(rank), d)
        raise(ValueError(message))

    # Make sure it's not a tuple but a list
    rank = list(rank)
    context = tl.context(input_tensor)

    # Initialization
    if rank[0] != 1:
        print('Provided rank[0] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[0] to 1.'.format(rank[0]))
        rank[0] = 1
    if rank[-1] != 1:
        print('Provided rank[-1] == {} but boundaring conditions dictatate rank[0] == rank[-1] == 1: setting rank[-1] to 1.'.format(rank[0]))


    # random selection of indices
    Jinit = [None] * d
    for k_Js in range(d-1):
        Jinit[k_Js] = []
        for i in range(rank[k_Js+1]):
            newidx = tuple( [ npr.choice(range(n[j])) for j in range(k_Js+1,d) ] )
            while newidx in Jinit[k_Js]:
                newidx = tuple( [ npr.choice(range(n[j])) for j in range(k_Js+1,d) ] )

            Jinit[k_Js].append(newidx)

    factor_old = [ tl.zeros((rank[k],n[k],rank[k+1])) for k in range(d) ]
    factor_new = [ tl.tensor(npr.random((rank[k],n[k],rank[k+1]))) for k in range(d) ]

    Js = Jinit
    it = 0
    while it < maxit and tl.norm(mps_to_tensor(factor_old)-mps_to_tensor(factor_new),2) > delta * tl.norm(mps_to_tensor(factor_new), 2):
        it += 1
        factor_old = factor_new
        factor_new = [None for i in range(d)]

        ######################################
        # left-to-right step
        ltor_fiber_list = []
        # list Is: list (d-1) of lists of left indices
        Is = [[()]]
        for k in range(d-1):
            (IT, fibers_list, Q, QsqInv) = left_right_ttcross_step(input_tensor, it, k, rank, Is, Js,  mv_eps, mv_maxit)
            ltor_fiber_list.extend( fibers_list )
            Is.append(IT)

        # end left-to-right step
        ###############################################

        ###############################################
        # right-to-left step
        rtol_fiber_list = []
        # list Js: list (d-1) of lists of right indices
        Js = [None] * d
        Js[-1] = [()]
        for k in range(d,1,-1):
            (JT, fibers_list, Q, QsqInv) = right_left_ttcross_step(input_tensor, it, k,rank, Is, Js, mv_eps, mv_maxit)
            rtol_fiber_list.extend( fibers_list )
            Js[k-2] = JT

            # Compute core
            try:
                factor_new[k-1] = tl.transpose(tl.dot(Q,QsqInv)).reshape( (rank[k-1], n[k-1], rank[k]) )
            except:
                raise(ValueError("The rank is too large compared to the size of the tensor. Try with small rank."))

        # Add the last core
        idx = (slice(None,None,None),) + tuple(zip(*Js[0]))

        C = input_tensor[ idx ]
        C = C.reshape(n[0], 1, rank[1])
        C = tl.transpose(C, (1,0,2) )

        factor_new[0] = C

        # end right-to-left step
        ################################################

        # Check that none of the previous iteration has already used the same fibers
        # (this indicates the presence of a loop).
        # If this is the case apply a random perturbation on one of the fibers
        # loop_detected = False
        # i = 0
        # while (not loop_detected) and i < len(input_tensor.rtol_fiber_lists)-1:
        #     loop_detected = all(map( operator.eq, input_tensor.rtol_fiber_lists[i], rtol_fiber_list )) \
        #         and all(map( operator.eq, input_tensor.ltor_fiber_lists[i], ltor_fiber_list ))
        #     i += 1
        #
        # if loop_detected:# and rtol_loop_detected:
        #     # If loop is detected, then an exception is raised
        #     # and the outer_ttcross will increase the rank
        #     input_tensor.Js = Js
        #     input_tensor.Is = Is
        #     # raise TTcrossLoopError('Loop detected!')
        # else:
        #     input_tensor.ltor_fiber_lists.append(ltor_fiber_list)
        #     input_tensor.rtol_fiber_lists.append(rtol_fiber_list)

    if it >= maxit:
        raise ValueError('Maximum number of iterations reached.')
    if tl.norm(mps_to_tensor(factor_old)-mps_to_tensor(factor_new),2) > delta * tl.norm(mps_to_tensor(factor_new), 2):
        raise ValueError('Low Rank Approximation algorithm did not converge.')

    return factor_new


def left_right_ttcross_step(input_tensor, it, k, rs, Is, Js, mv_eps, mv_maxit):
    """ Compute one step of left-right sweep of ttcross.

    :param int it: the actual ttcross iteration
    :param int k: the actual sweep iteration
    :param list rs: list of upper ranks (d)
    :param list Is: list (d-1) of lists of left indices
    :param list Js: list (d-1) of lists of right indices
    :param float mv_eps: MaxVol accuracy
    :param int mv_maxit: maximum number of iterations for MaxVol

    :returns: tuple containing: ``(IT,fibers_list,Q,QsqInv)``, the list of new indices, the used fibers, the Q matrix and the inverse of the maxvol submatrix
    """

    n = input_tensor.shape
    d = len(n)
    fibers_list = []

    # Extract fibers
    for i in range(rs[k]):
        for j in range(rs[k+1]):
            fiber = Is[k][i] + (slice(None,None,None),) + Js[k][j]
            fibers_list.append(fiber)
    if k == 0:      # Is[k] will be empty
        idx = (slice(None,None,None),) + tuple(zip(*Js[k]))
    else:
        idx = [ [] for i in range(d) ]
        for lidx in Is[k]:
            for ridx in Js[k]:
                for j,jj in enumerate(lidx): idx[j].append(jj)
                for j,jj in enumerate(ridx): idx[len(lidx)+1+j].append(jj)
        idx[k] = slice(None,None,None)
        idx = tuple(idx)

    # print(idx)
    C = input_tensor[ idx]
    # print(C)

    if k == 0:
        C = C.reshape(n[k], rs[k], rs[k+1])
        C = tl.transpose(C, (1,0,2) )
    else:
        C = C.reshape(rs[k], rs[k+1], n[k])
        C = tl.transpose(C, (0,2,1) )

    C = C.reshape(( rs[k] * n[k], rs[k+1] ))

    # Compute QR decomposition
    (Q,R) = tl.qr(C)

    # Maxvol
    (I,QsqInv,it) = maxvol(Q,mv_eps,mv_maxit)

    # Retrive indices in folded tensor
    IC = [ idxfold( [rs[k],n[k]], idx ) for idx in I ] # First retrive idx in folded C
    IT = [ Is[k][ic[0]] + (ic[1],) for ic in IC ] # Then reconstruct the idx in the tensor

    return (IT, fibers_list, Q, QsqInv)

def right_left_ttcross_step(input_tensor, it, k, rs, Is, Js, mv_eps, mv_maxit):
    """ Compute one step of right-left sweep of ttcross.

    :param int it: the actual ttcross iteration
    :param int k: the actual sweep iteration
    :param list rs: list of upper ranks (d)
    :param list Is: list (d-1) of lists of left indices
    :param list Js: list (d-1) of lists of right indices
    :param float mv_eps: MaxVol accuracy
    :param int mv_maxit: maximum number of iterations for MaxVol

    :returns: tuple containing: ``(JT,fibers_list,Q,QsqInv)``, the list of new indices, the used fibers, the Q matrix and the inverse of the maxvol submatrix
    """

    n = input_tensor.shape
    d = len(n)
    fibers_list = []

    # Extract fibers
    for i in range(rs[k-1]):
        for j in range(rs[k]):
            fiber = Is[k-1][i] + (slice(None,None,None),) + Js[k-1][j]
            fibers_list.append(fiber)

    if k == d:      # Is[k] will be empty
        idx = tuple(zip(*Is[k-1])) + (slice(None,None,None),)
    else:
        idx = [ [] for i in range(d) ]
        for lidx in Is[k-1]:
            for ridx in Js[k-1]:
                for j,jj in enumerate(lidx): idx[j].append(jj)
                for j,jj in enumerate(ridx): idx[len(lidx)+1+j].append(jj)
        idx[k-1] = slice(None,None,None)
        idx = tuple(idx)

    C = input_tensor[ idx]
    C = C.reshape(rs[k-1], rs[k], n[k-1])
    C = tl.transpose(C, (0,2,1) )

    C = C.reshape( (rs[k-1],n[k-1]*rs[k]) )
    C = tl.transpose(C)

    # Compute QR decomposition
    (Q,R) = tl.qr(C)
    # Maxvol
    (J,QsqInv,it) = maxvol(Q,mv_eps,mv_maxit)

    # Retrive indices in folded tensor
    JC = [ idxfold( [n[k-1],rs[k]], idx ) for idx in J ] # First retrive idx in folded C
    JT = [  (jc[0],) + Js[k-1][jc[1]] for jc in JC ] # Then reconstruct the idx in the tensor

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

    :returns: ``(I,AsqInv,it)`` where ``I`` is the list or rows of A forming the matrix with maximal volume, ``AsqInv`` is the inverse of the matrix with maximal volume and ``it`` is the number of iterations to convergence
    """

    (n,r) = A.shape

    if r>n :
        raise TypeError("maxvol: A(nxr) must be a thin matrix, i.e. n>=r")

    # Find an arbitrary non-singular rxr matrix in A
    (P,L,U) = scla.lu(A)

    # Check singularity
    if tl.min(tl.abs(tl.tensor(np.diag(U)))) < np.spacing(1):
        raise ValueError("maxvol: Matrix A is singular")

    # Reorder A so that the non-singular matrix is on top
    I = tl.arange(n) # set of swapping indices
    I = tl.dot(tl.tensor(P.T),I)
    I = tl.int(I)

    # Compute inverse of Asq: Asq^-1 = (PLU)^-1
    LU = L[:r,:r] - np.eye(r) + U
    AsqInv = scla.lu_solve((LU,tl.arange(r)), np.eye(r))
    AsqInv = tl.tensor(AsqInv)
    # Compute B
    B = tl.dot(A[I,:],tl.tensor(AsqInv))


    maxrow = np.argmax(tl.abs(B),axis=0)[0]
    maxcol = np.argmax(tl.abs(B),axis=1)[0]
    maxB = tl.abs(B)[maxrow,maxcol]
    # print(B,maxrow,maxcol,maxB)

    it = 0
    eps = 1.+ delta
    while it < maxit and maxB > eps:
        it += 1

        # Update AsqInv
        q = tl.zeros((r,1))
        q[maxcol] = 1.
        vT = A[[I[maxrow],],:] - A[[I[maxcol],],:]
        # Eq (8) in "How to find a good submatrix"
        AsqInv -= tl.dot(tl.dot(AsqInv,q), tl.dot(vT,AsqInv)) / (1. + tl.dot(vT,tl.dot(AsqInv,q)))

        # Update B using Sherman-Woodbury-Morrison formula
        Bj = B[:,[maxcol,]]
        # Bj[maxcol,0] -= 1.
        Bj[maxrow,0] += 1.
        Bi = B[[maxrow,],:]
        Bi[0,maxcol] -= 1.
        B[r:,:] -= tl.dot(Bj[r:],Bi)/B[maxrow,maxcol]

        # Update index of maxvol matrix I
        tmp = I[maxcol]
        I[maxcol] = I[maxrow]
        I[maxrow] = tmp

        # Find new maximum in B
        maxrow = np.argmax(tl.abs(B),axis=0)[0]
        maxcol = np.argmax(tl.abs(B),axis=1)[0]
        maxB = tl.abs(B)[maxrow,maxcol]

    if maxB > eps:
        raise ValueError('Maxvol algorithm did not converge.')

    # Return max-vol submatrix Asq
    return ([np.asscalar(i) for _,i in np.ndenumerate(I[:r])],AsqInv,it)
