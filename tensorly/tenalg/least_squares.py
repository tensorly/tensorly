from .. import backend as T


def solve_least_squares(A, B):
    # solve ||B-AX||^2 (AX = B -> X = A^+ @ B), with A^+: pseudo inverse
    X = T.pinv(A) @ B
    return X
