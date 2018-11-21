import tensorly as tl
import numpy as np

# Author : Jeremy Cohen, copied from Nicolas Gillis code for HALS on Matlab


def nnlsHALSupdt(M, U, V, maxiter = 500):
    """ Computes an approximate solution of the following nonnegative least 
     squares problem (NNLS)  

               min_{V >= 0} ||M-UV||_F^2 
     
     with an exact block-coordinate descent scheme. 

     See N. Gillis and F. Glineur, Accelerated Multiplicative Updates and 
     Hierarchical ALS Algorithms for Nonnegative Matrix Factorization, 
     Neural Computation 24 (4): 1085-1105, 2012.
     

     ****** Input ******
       M  : m-by-n matrix 
       U  : m-by-r matrix
       V  : r-by-n initialization matrix 
            default: one non-zero entry per column corresponding to the 
            clostest column of U of the corresponding column of M 
       maxiter: upper bound on the number of iterations (default=500).

       *Remark. M, U and V are not required to be nonnegative. 

     ****** Output ******
       V  : an r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2
    """
    m, n = tl.shape(M)
    m, r = tl.shape(U)
    UtU  = tl.dot(tl.transpose(U),U)
    UtM  = tl.dot(tl.transpose(U),M)

    if not V: #checks if V is empty
        V = tl.solve(U,M) # Least squares        
        V[V<0] = 0
        # Scaling
        alpha = 
