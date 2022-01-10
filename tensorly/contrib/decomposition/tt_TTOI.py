
import numpy as np
from math import prod
import tensorly as tl
from tensorly.tt_tensor import tt_to_tensor, validate_tt_rank
import warnings



def sequential_prod_reshape(tensor, M_arr, direction):
    """ Perform sequential multiplication and reshaping

    Parameters
    ----------
    tensor: 2d array
        matrix to be multiplied by a list of matrices
        tensor is a (p_1p_2...p_{d-1})*p_d matrix or a p_1*(p_2...p_d) matrix
    M_arr : list 
        a list of 2d arrays to multiply tensor
        len(M_arr) <= d-1
        If direction == "left", M_arr{i} is a r_{i-1}p_i*r_i matrix
        If direction == "right", M_arr{i} is a r_{d-i+1}p_{d-i+1}*r_{d-i} matrix 
    direction : string
        direction of the multiplication
    Returns
    -------
    tensor_prod : 2d array
        product matrix
        If direction == "left" and M_arr has k elements, then tensor_prod is of dimension r_k*(p_{k+1}...p_d) 
        If direction == "right" and M_arr has d-k elements, then tensor_prod is of dimension (p_1p_2...p_k)*r_k
    """ 

    if direction == "left":
        tensor_prod = tl.matmul(tl.transpose(M_arr[0]),tensor)
        for i in range(1,len(M_arr)):
            tensor_prod = tl.reshape(tensor_prod,(tl.shape(M_arr[i])[0],-1))
            tensor_prod = tl.matmul(tl.transpose(M_arr[i]),tensor_prod)   
    else:
        tensor_prod = tl.matmul(tensor,M_arr[0])
        for i in range(1,len(M_arr)):
            tensor_prod = tl.reshape(tensor_prod,(-1,tl.shape(M_arr[i])[0]))
            tensor_prod = tl.matmul(tensor_prod,M_arr[i])
    return tensor_prod



def TTOI(data_tensor, rank, niter, trajectory = False, **context):
    """ Perform tensor-train orthogonal iteration (TTOI) for tensor train decomposition
    Reference paper: Zhou Y, Zhang AR, Zheng L, Wang Y. "Optimal high-order tensor svd via tensor-train orthogonal iteration."

    Parameters
    ----------
    data_tensor: ndarray
        observed tensor data
    rank : tuplee
        rank of the TT decomposition
        must verify rank[0] == rank[-1] == 1 (boundary conditions)
        and len(rank) == len(tl.shape(data_tensor))+1
    niter : int
        number of iterations
    trajectory : bool, optional, default is False
        if True, the output of each iteration of TTOI is returned: niter outputs
        otherwise, the output of the last iteration is returned
    context : dict
        context in which to create the tensor
    Returns
    -------
    factors_list or factors : list of niter tensors or one tensor
        * niter tensors (if `trajectory` is True) : each list contains the output of each iteration, one full tensor and list of tensor factoros
        * one tensor (otherwise): output of the last iteration, one full tensor and list of tensor factors
    full_tensor_list or full_tensor : list of niter tensors or one tensor
        * niter tensors (if `trajectory` is True) : each list contains the output of each iteration, one full tensor and list of tensor factoros
        * one tensor (otherwise): output of the last iteration, one full tensor and list of tensor factors
    """
    shape = tl.shape(data_tensor)
    n_dim = len(shape) 

    rank = validate_tt_rank(shape, rank)

    # Make sure it's not a tuple but a list
    rank = list(rank)

    # Initialization of tensor train factors
    # U_arr will be a list of niter lists, each list being estimated left singular spaces at each iteration
    U_arr = list();
    # V_arr will be a list of niter lists, each list being estimated right singular spaces at each iteration
    V_arr = list();

    tensor_arr = list();
    # tensor_arr contains the sequential unfoldings (specific for tensor train, different from the common definition of unfolding) of data_tensor
    for i in range(1,n_dim):
        tensor_arr.append(tl.reshape(data_tensor,(prod(shape[0:i]),prod(shape[i:n_dim]))))
    if trajectory:
        factors_list = list()
        full_tensor_list = list()

    # perform TTOI for niter iterations
    for n in range(niter):
        # first perform forward update
        # initialize U_arr[n] and R_tilde_arr (sequential unfolding of data_tensor multiplied by U_arr sequentially on the left, useful for backward update to obtain V_arr[n])
        U_arr.append(list())
        R_tilde_arr = list()

        # estimate the first left singular spaces
        # Here, R_tmp is the first sequential unfolding compressed on the right by previous updated V_arr (if exists)
        if n == 0:
            R_tmp = tensor_arr[0]
        else:
            R_tmp = sequential_prod_reshape(tensor_arr[n_dim-2],V_arr[n-1],"right")
        U_tmp = tl.partial_svd(R_tmp,rank[1])[0]
        U_arr[n].append(U_tmp)

        # estimate the 2nd to (d-1)th left singular spaces
        for k in range(n_dim-2):
            # compress the (k+2)th sequential unfolding of data_tensor from the left
            R_tmp_l = sequential_prod_reshape(tensor_arr[0],U_arr[n][0:(k+1)],"left")
            # R_tmp_l will be useful for backward update
            R_tilde_arr.append(R_tmp_l)
            R_tmp_l = tl.reshape(R_tmp_l,(rank[k+1]*shape[k+1],prod(shape[(k+2):n_dim])))

            # compress the (k+2)th sequential unfolding of data_tensor from the right (if n>0)
            if n == 0:
                R_tmp = R_tmp_l
            else:
                R_tmp = sequential_prod_reshape(tl.reshape(R_tmp_l,(rank[k+1]*prod(shape[k+1:(n_dim-1)]),shape[n_dim-1])),V_arr[n-1][0:(n_dim-k-2)],"right")
            U_tmp = tl.partial_svd(R_tmp,rank[k+2])[0]
            U_arr[n].append(U_tmp)

        # forward update is done; output the final residual
        R_tilde_arr.append(sequential_prod_reshape(tensor_arr[0],U_arr[n],"left"))
        if trajectory:
            factors_list.append(list())
            for k in range(n_dim-1):
                factors_list[2*n].append(tl.tensor(tl.reshape(U_arr[n][k],(rank[k],shape[k],rank[k+1])),**context))
            factors_list[2*n].append(tl.tensor(tl.reshape(R_tilde_arr[n_dim-2],(rank[n_dim-1],shape[n_dim-1],rank[n_dim])),**context))
            full_tensor_list.append(tl.tensor(tt_to_tensor(factors_list[2*n]),**context))


        # perform backward update
        # initialize V_arr
        V_arr.append(list())
        V_tmp = tl.transpose(tl.partial_svd(R_tilde_arr[n_dim-2],rank[n_dim-1])[2])
        V_arr[n].append(V_tmp)

        # estimate the 2nd to (d-1)th right singular spaces
        for k in range(n_dim-2):
            # compress R_tilde_arr from the right
            R_tmp_r = sequential_prod_reshape(tl.reshape(R_tilde_arr[n_dim-k-3],(-1,shape[n_dim-1])),V_arr[n][0:(k+1)],"right")
            R_tmp = tl.reshape(R_tmp_r,(rank[n_dim-k-2],shape[n_dim-k-2]*rank[n_dim-k-1]))
            V_tmp = tl.transpose(tl.partial_svd(R_tmp,rank[n_dim-k-2])[2])
            V_arr[n].append(V_tmp)

        Residual_right = sequential_prod_reshape(tensor_arr[n_dim-2],V_arr[n],"right")
        if trajectory:
            factors_list.append(list())
            factors_list[2*n+1].append(tl.tensor(tl.reshape(Residual_right,(rank[0],shape[0],rank[1])),**context))
            for k in range(n_dim-1):
                factors_list[2*n+1].append(tl.tensor(tl.reshape(tl.transpose(V_arr[n][n_dim-k-2]),(rank[k+1],shape[k+1],rank[k+2])),**context))
            full_tensor_list.append(tl.tensor(tt_to_tensor(factors_list[2*n+1]),**context))
        elif n == niter-1:
            factors = list()
            factors.append(tl.tensor(tl.reshape(Residual_right,(rank[0],shape[0],rank[1])),**context))
            for k in range(n_dim-1):
                factors.append(tl.tensor(tl.reshape(tl.transpose(V_arr[n][n_dim-k-2]),(rank[k+1],shape[k+1],rank[k+2])),**context))
            full_tensor = tl.tensor(tt_to_tensor(factors_list[2*n+1]),**context)

    # return final results
    if trajectory:
        return factors_list, full_tensor_list
    else:
        return factors, full_tensor

