import torch 


def cg(op: callable, x, rhs, n_iter: int = 5):
    """
    Batched version of conjugate gradient (CG) descent, running for a fixed number of iterations. 
    The batching is done over the second dimension, i.e., we assume a dimension of x as (dimension, batch)
    
    Solves the (square) system of equations: B x = y 
    
    A note on the usage: 
        Most of the times we apply CG to the normal equations
            A^T A x = A^T y 
        or with Tikhonov
            (A^T A + gamma I) x = A^T y

    Arguments:
        op: implementation of the operator B 
        x: initialisation
        rhs: right hand side y
        n_iter: total number of iterations
    """
    # n x batch
    r = op(x)
    r = rhs - r
    p = r
    d = torch.zeros_like(x)
    
    sqnorm_r_old = torch.sum(r*r, dim=0)
    for _ in range(n_iter):
        
        d = op(p)
        
        inner_p_d = (p * d).sum(dim=0) 

        alpha = sqnorm_r_old / inner_p_d
        x = x + alpha[None,:]*p # x = x + alpha*p
        r = r - alpha[None,:]*d # r = r - alpha*d

        sqnorm_r_new = torch.sum(r * r, dim=0)

        beta = sqnorm_r_new / sqnorm_r_old
        sqnorm_r_old = sqnorm_r_new

        p = r + beta[None,:]*p # p = r + b * p

    return x 
