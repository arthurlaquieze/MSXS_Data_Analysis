"""
@author: gurol
    
"""
import numpy as np

def pcg(A,x0,b,F,maxit,tol):
    """ 
    Preconditioned Conjugate Gradient Algorithm
        pcg solves the symmetric positive definite linear system 
            A x  = b 
        with a preconditioner P
        A     : n-by-n symmetric and positive definite matrix
        b     : n dimensional right hand side vector
        F     : n-by-n preconditioner matrix (an approximation to inv(A))
        maxit : maximum number of iterations
        tol   : error tolerance on the residual 
    """
    flag = 0
    verbose = False
    x = np.copy(x0)
    X = x
    r = b - A.dot(x)
    nrmb = np.linalg.norm(b)
    error = np.linalg.norm(r)/nrmb

    if error < tol:
        return x, error, 0, flag

    for i in range(maxit):
        z = F.dot(r)
        rho = np.dot(r.T,z) 
        if verbose:
            print("Iter ", i, " rho :", rho)
        if i > 0:
            beta = rho / rho_prev
            p = z + beta*p
        else:
            p = np.copy(z)
        q = A.dot(p)
        curvature = np.dot(p.T, q)
        if verbose:
            print("   curvature :", curvature)
        alpha = rho / curvature
        x = x + alpha*p
        X = np.concatenate((X,x))
        r = r - alpha*q
        error =  np.linalg.norm(r)/nrmb
        if error < tol:
            return X, error, i, flag   
        rho_prev = rho    

    if error > tol: 
        #print("No convergence")
        flag = 1
    return X, error, i, flag

def Bcg(B,HtRinvH,x0,b,maxit,tol):
    """ 
    B-Right Preconditioned Conjugate Gradient Algorithm
        Bcg_right solves the linear system 
             (I + HtRinvH B) v = b
        with an inner product < , >_B
        HtRinvH : n-by-n symmetric and positive definite matrix
        B       : n-by-n symmetric and (positive definite matrix)!
        b       : n dimensional right hand side vector
        maxit   : maximum number of iterations
        tol     : error tolerance on the residual 
    """
    flag = 0
    verbose = False
    x = np.copy(x0)
    xh = np.copy(x0)
    X = x
    #r = b - np.matmul(A,x)
    #r  = b - x - HtRinvH.dot(B.dot(x))
    r = b
    z  = B.dot(r)
    h  = r

    nrmb = (np.dot(b.transpose(), B.dot(b)))**0.5
    error = ((np.dot(r.transpose(), z))**0.5)/nrmb
    if error < tol:
        return x, error, 0, flag

    for i in range(maxit):
        z = B.dot(r)
        rho = np.dot(r.transpose(),z) 
        if verbose:
            print("Iter ", i, " rho :", rho)
        if i > 0:
            beta = rho / rho_prev
            p = z + beta*p
            h = r + beta*h
        else:
            p = z
            h = r
        #q = h + HtRinvH.dot(p)
        q = h + HtRinvH.dot(p)
        curvature = np.dot(p.T, q)
        if verbose:
            print("   curvature :", curvature)
        if curvature < 0:
            print('negative curvature')    
        alpha = rho / curvature
        x = x + alpha*p
        X = np.concatenate((X,x))
        r = r - alpha*q
        error =  (np.dot(r.T, B.dot(r)))**0.5/nrmb
        if error < tol:
            return X, error, i, flag   
        rho_prev = rho    

    if error > tol: 
        #print("No convergence")
        flag = 1
    return X, error, i, flag    


if __name__ == '__main__':
    n = 10
    A = np.random.rand(n,n)
    A = np.matmul(A, A.transpose())
    P = np.eye(n)
    b = np.random.rand(n,1)
    x0 = np.zeros_like(b)
    xstar = np.linalg.solve(A, b)
    maxit = 20
    tol   = 1e-10
    print(" > Solution with Conjugate Gradient Algorithm")
    [cg_sol, error, iter, flag] = pcg(A,x0,b,P,maxit,tol)
    print("|| xstar - cg_sol || : ", np.linalg.norm(xstar-cg_sol[-n:]),'\n')    

    # Data Assimilation Set-Up
    #H matrix
    m = 6
    Hmat = np.eye(n)
    inds=np.random.permutation(m)
    Hmat = Hmat[:,inds]
    Hmat = Hmat.transpose()
    #R matrix
    Imat = np.eye(n)
    sigma_r = 0.3
    Rinv = sigma_r*np.eye(m)
    RinvH  = np.matmul(Rinv, Hmat)
    #Hessian 
    HtRinvH = np.matmul(Hmat.transpose(), RinvH)
    
    #B matrix
    B12 = np.random.rand(n,n)
    B = np.dot(B12, B12.transpose())
    A = np.linalg.inv(B) + HtRinvH

    xstar = np.linalg.solve(A, b)
    print(" > Solution with Conjugate Gradient Algorithm")
    [cg_sol, error, iter, flag] = pcg(A,x0,b,np.linalg.inv(B),maxit,tol)
    print("|| xstar - cg_sol || : ", np.linalg.norm(xstar-cg_sol[-n:]),'\n') 
    print(" > Solution with B - Conjugate Gradient Algorithm")
    [cgright_sol, error, iter, flag] = Bcg(B,HtRinvH,x0,b,maxit,tol)
    print("|| xstar - cgright_sol || : ", np.linalg.norm(xstar-cgright_sol[-n:]))


    
    
