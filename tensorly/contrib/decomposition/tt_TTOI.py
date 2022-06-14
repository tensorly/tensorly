
import numpy as np
from math import prod
import tensorly as tl
from tensorly.tt_tensor import tt_to_tensor, validate_tt_rank
import warnings
from tensorly.decomposition._base_decomposition import DecompositionMixin



def sequential_prod(tensor_prod, multiplier_list, direction):
    """ Perform sequential multiplication and reshaping

    Parameters
    ----------
    tensor_prod: ndrray
        the tensor to be multiplied by a list of tensors
        tensor_prod is of dimension r_0*p_1*p_2*...*p_d*r_{d-1}
    multiplier_list : list 
        a list of tensors to multiply the tensor
        len(multiplier_list) <= d-1
        If direction == "left", multiplier_list[i] is of shape (r_i, p_{i+1}, r_{i+1})
        If direction == "right", multiplier_list[i] is of shape (r_{d-i}, p_{d-i}, r_{d-i-1})
    direction : string
        direction of the multiplication
    Returns
    -------
    tensor_prod : ndarray
        product tensor
        If direction == "left" and `multiplier_list` contains N tensors, then `tensor_prod` is of dimension r_N*p_{N+1}*...*p_d*r_{d-1}
        If direction == "right" and `multiplier_list` contains d-N tensors, then `tensor_prod` is of dimension r_0*p_1*...*p_N*r_N
    """ 

    if direction == "left":
        for i in range(len(multiplier_list)):
            tensor_prod = tl.tensordot(multiplier_list[i], tensor_prod, axes = ([0, 1], [0, 1]))   
    else:
        for i in range(len(multiplier_list)):
            tensor_prod = tl.tensordot(tensor_prod, multiplier_list[i], axes = ([tl.ndim(tensor_prod)-1, tl.ndim(tensor_prod)-2], [0, 1]))
    return tensor_prod



def tensor_train_OI(data_tensor, rank, n_iter = 1, trajectory = False, return_errors = True, **context):
    """ Perform tensor-train orthogonal iteration (TTOI) for tensor train decomposition
    Reference paper: Zhou Y, Zhang AR, Zheng L, Wang Y. "Optimal high-order tensor svd via tensor-train orthogonal iteration."

    Parameters
    ----------
    data_tensor: tl.tensor
        observed tensor data
    rank : tuple
        rank of the TT decomposition
        must verify rank[0] == rank[-1] == 1 (boundary conditions)
        and len(rank) == len(tl.shape(data_tensor))+1
    n_iter : int
        half the number of iterations
    trajectory : bool, optional, default is False
        if True, the output of each iteration of TTOI is returned: 2*n_iter outputs
        otherwise, only the output of the last iteration is returned
    return_errors : bool, optional, default is True
        if True, the approximation/reconstruction error of each iteration of TTOI is returned: 2*n_iter outputs
    Returns
    -------
    factors : list of n_iter tensors or one tensor
    * n_iter tensors (if `trajectory` is True) : each list contains the output of each iteration, one full tensor and list of tensor factors
    * one tensor (otherwise): output of the last iteration, one full tensor and list of tensor factors 
    full_tensor : list of n_iter tensors or one tensor
    * n_iter tensors (if `trajectory` is True) : each list contains the output of each iteration, one full tensor and list of tensor factoros
    * one tensor (otherwise): output of the last iteration, one full tensor and list of tensor factors 
    """
    context = tl.context(data_tensor)
    shape = tl.shape(data_tensor)
    n_dim = len(shape) 

    rank = validate_tt_rank(shape, rank)

    # Make sure it's not a tuple but a list
    rank = list(rank)
    
    # Add two one-dimensional mode to data_tensor
    data_tensor_extended = tl.reshape(data_tensor,(1, ) + shape + (1, ))


    if trajectory:
        factors = list()
        full_tensor = list()
        
    if return_errors:
        error_list = list()
        
    # perform TTOI for n_iter iterations
    for n in range(n_iter):
        # first perform forward update
        # U_arr will be a list including estimated left singular spaces at the current iteration
        U_arr = list()
        # initialize R_tilde_arr (sequential unfolding of data_tensor multiplied by U_arr sequentially on the left, useful for backward update to obtain V_arr)
        R_tilde_arr = list()

        # estimate the first left singular spaces
        # Here, R_tmp is the first sequential unfolding compressed on the right by previous updated V_arr (if exists)
        R_tmp_l = data_tensor_extended
        if n == 0:
            R_tmp = R_tmp_l
        else:
            R_tmp = sequential_prod(R_tmp_l,V_arr,"right")
        U_tmp = tl.partial_svd(tl.reshape(R_tmp,(shape[0],-1)),rank[1])[0]
        U_arr.append(tl.reshape(U_tmp,(rank[0],shape[0],rank[1])))

        # estimate the 2nd to (d-1)th left singular spaces
        for k in range(n_dim-2):
            # compress the (k+2)th sequential unfolding of data_tensor from the left
            R_tmp_l = sequential_prod(R_tmp_l,[U_arr[k]],"left")
            # R_tmp_l will be useful for backward update
            R_tilde_arr.append(R_tmp_l)
            
            # compress the (k+2)th sequential unfolding of data_tensor from the right (if n>0)
            if n == 0:
                R_tmp = R_tmp_l
            else:
                R_tmp = sequential_prod(R_tmp_l,V_arr[0:(n_dim-k-2)],"right")
            U_tmp = tl.partial_svd(tl.reshape(R_tmp,(rank[k+1]*shape[k+1],-1)),rank[k+2])[0]
            U_arr.append(tl.reshape(U_tmp,(rank[k+1],shape[k+1],rank[k+2])))

        # forward update is done; output the final residual
        R_tilde_arr.append(sequential_prod(R_tmp_l,[U_arr[n_dim-2]],"left"))
        if trajectory or return_errors:
            factors_list_tmp = list()
            for k in range(n_dim-1):
                factors_list_tmp.append(tl.tensor(U_arr[k],**context))
            factors_list_tmp.append(tl.tensor(R_tilde_arr[n_dim-2],**context))
            full_tensor_tmp = tl.tensor(tt_to_tensor(factors_list_tmp),**context)
            if return_errors:
                error_list.append(tl.norm(full_tensor_tmp-data_tensor,2))
            if trajectory:
                factors.append(factors_list_tmp)
                full_tensor.append(full_tensor_tmp)



        # perform backward update
        # initialize V_arr: V_arr will be a list of estimated right singular spaces at the current or previous iteration
        V_arr = list()
        V_tmp = tl.transpose(tl.partial_svd(tl.reshape(R_tilde_arr[n_dim-2],(rank[n_dim-1],shape[n_dim-1])),rank[n_dim-1])[2])
        V_arr.append(tl.reshape(V_tmp,(rank[n_dim],shape[n_dim-1],rank[n_dim-1])))

        # estimate the 2nd to (d-1)th right singular spaces
        for k in range(n_dim-2):
            # compress R_tilde_arr from the right
            R_tmp_r = sequential_prod(R_tilde_arr[n_dim-k-3],V_arr[0:(k+1)],"right")
            V_tmp = tl.transpose(tl.partial_svd(tl.reshape(R_tmp_r,(rank[n_dim-k-2],shape[n_dim-k-2]*rank[n_dim-k-1])),rank[n_dim-k-2])[2])
            V_arr.append(tl.reshape(V_tmp,(rank[n_dim-k-1],shape[n_dim-k-2],rank[n_dim-k-2])))

        Residual_right = sequential_prod(data_tensor_extended,V_arr,"right")
        if trajectory or return_errors or n==n_iter-1:
            factors_list_tmp = list()
            factors_list_tmp.append(tl.tensor(Residual_right,**context))
            for k in range(n_dim-1):
                factors_list_tmp.append(tl.tensor(tl.transpose(V_arr[n_dim-k-2]),**context))
            full_tensor_tmp = tl.tensor(tt_to_tensor(factors_list_tmp),**context)
            if return_errors:
                error_list.append(tl.norm(full_tensor_tmp-data_tensor,2))
            if trajectory:
                factors.append(factors_list_tmp)
                full_tensor.append(full_tensor_tmp)
            if n == n_iter-1:
                factors = factors_list_tmp
                full_tensor = full_tensor_tmp

    # return final results
    if return_errors:
        return factors, full_tensor, error_list
    else:
        return factors, full_tensor


class TensorTrain_OI(DecompositionMixin):

    def __init__(self, rank, n_iter, trajectory, return_errors):
        self.rank = rank
        self.n_iter = n_iter
        self.trajectory = trajectory
        self.return_errors = return_errors

    def fit_transform(self, tensor):
        self.decomposition_ = tensor_train_OI(tensor, rank=self.rank, n_iter=self.n_iter,
                                              trajectory=self.trajectory, return_errors=self.return_errors)
        return self.decomposition_
