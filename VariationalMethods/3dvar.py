# Variational Data Assimilation
# Programming: Selime Gurol (CERFACS), 2021

import sys

sys.path.append("..")

from numpy.core.numeric import zeros_like
from numpy import (
    round,
    shape,
    copy,
    dot,  # Matrix-matrix or matrix-vector product
    eye,  # To generate an identity matrix
    ones,  # To generate an array full of ones
    zeros,  # To generate an array full of zeros
    linspace,
)  # To get space and time position indices for observations
import numpy as np
from numpy.linalg import (
    inv,  # To invert a matrix
    norm,
)  # To compute the Euclidean norm
from numpy.random import randn  # To generate samples from a normalized Gaussian
import matplotlib.pyplot as plt  # To plot a graph
from operators import Hessian3dVar, obs_operator, Rmatrix, Bmatrix
from operators import Precond
from Model.models import lorenz95
from solvers import pcg


def funcval(x):
    eo = y - obs.hop(x)
    eb = x - xb
    J = eb.dot(B.invdot(eb)) + eo.dot(R.invdot(eo))
    return J


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (1) Initialization
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 40  # state space dimension

# Model class initialization
dt = 0.025  # time step for 4th order Runge-Kutta
F = 8  # Forcing term
model = lorenz95(F, dt)

# Observation class initialization
sigmaR = 0.2  # observation error std
total_space_obs = 20  # total number of observations at fixed time
space_inds_obs = round(linspace(0, n, total_space_obs, endpoint=False)).astype(
    int
)  # observation locations in the space
obs = obs_operator(sigmaR, space_inds_obs, n)
R = Rmatrix(sigmaR)

# Background class initialization
sigmaB = 0.8  # background error std
# B = Bmatrix(sigmaB,'diagonal')
B = Bmatrix(sigmaB, "diffusion", D=10, M=4)


# Minimization initialization
max_outer = 5  # number of maximum outer loops
max_inner = 20  # number of maximum inner loops
tol = 1e-6  # tolerance for the inner loop
tol_grad = 1e-6  # tolerance for the outer loop
In = eye(n)
F = Precond(B)  # Define the preconditioner

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (2) Generate the truth
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xt = 3.0 * ones(n) + randn(n)  # the true state
xt = model.traj(xt, 5000)  # spin-up


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (3) Generate the background
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# xb = xt + sigmaB*randn(n)
# The square root of B can be used to create correlated errors
xb = xt + B.sqrtdot(randn(n))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (4) Generate the observations
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# y = obs.hop(xt) + np.dot(sigmaR, np.random.randn(n))
y = obs.hop(xt) + sigmaR * np.random.randn(n)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (5) Variational data assimilation
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter_outer = 0
xa = copy(xb)  # Choose the initial vector
print("")
print("iter", "         f(x)", "          ||grad(x)||")
while iter_outer < max_outer:  # Gauss-Newton loop
    A = Hessian3dVar(obs, R, B)  # Define the Hessian matrix (Binv + HtRinvH)
    # TO DO                      # Complete Hessian3dVar in operators.py

    d = y - obs.hop(xa)  # MISFIT
    # b =  # Right hand side vector (Binv(xb - xa) + Ht*Rinv*d)
    b = B.invdot(xb - xa) + obs.hopT(R.invdot(d))
    print("{:<9d}{:<20.2f}{:<9.2f}".format(iter_outer, funcval(xa), norm(b)))
    if norm(b) < tol_grad:
        break
    # Calculate the increment dx such that
    # (Binv + HtRinvH) dx = Binv(xb - x) + Ht Rinv d
    # Solve the linear system by using an iterative solver PCG
    dxs, error, i, flag = pcg(A, zeros_like(xa), b, F, max_inner, tol)
    xa += dxs[-n:]
    iter_outer += 1

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (5) Diagnostics
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print("")
print("||xt - xb||_2 / ||xt||_2 = ", norm(xt - xb) / norm(xt))
print("||xt - xa||_2 / ||xt||_2 = ", norm(xt - xa) / norm(xt))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (6) Plots
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xrange = range(0, n)
fig, ax = plt.subplots()
ax.plot(xrange, xt, "-k", label="Truth")
ax.plot(xrange, xb, "-.b", label="Background")
ax.plot(space_inds_obs, y, "og", label="Observations")
ax.plot(xrange, xa, "-r", label="Analysis")
leg = ax.legend()
plt.xlabel("x-coordinate")
plt.ylabel("Temperature")
plt.show()
