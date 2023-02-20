# Variational Data Assimilation
# Programming: Selime Gurol (CERFACS), 2021

import sys
import math
from numpy.lib.function_base import append

sys.path.append("..")

from numpy.core.numeric import zeros_like
from numpy import (
    round,
    shape,
    copy,
    zeros,
    dot,  # Matrix-matrix or matrix-vector product
    eye,  # To generate an identity matrix
    ones,  # To generate an array full of ones
    random,
    concatenate,
    linspace,
)  # To get space and time position indices for observations
from numpy.linalg import (
    inv,  # To invert a matrix
    norm,
)  # To compute the Euclidean norm
from numpy.random import randn  # To generate samples from a normalized Gaussian
import matplotlib.pyplot as plt  # To plot a graph
from operators import obs_operator, Rmatrix, Bmatrix
from operators import Hessian4dVar, Precond
from Model.models import lorenz95
from solvers import pcg, Bcg


def nonlinear_funcval(x):
    eo = y - obs.gop(x)
    eb = x - xb
    J = eb.dot(B.invdot(eb)) + eo.dot(R.invdot(eo))
    return J


def quadratic_funcval(x, dx):
    eo = obs.gop(x) - y + obs.tlm_gop(x, dx)
    eb = x - xb + dx
    J = eb.dot(B.invdot(eb)) + eo.dot(R.invdot(eo))
    return J


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (1) Initialization
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
random.seed(1)
n = 100  # state space dimension
nt = 10  # number of time steps

# Model class initialization
dt = 0.025  # time step for 4th order Runge-Kutta
F = 8  # Forcing term
model = lorenz95(F, dt)

# Observation class initialization
sigmaR = 1e-4  # observation error std
total_space_obs = 10  # total number of observations at fixed time
total_time_obs = 5  # total number of observations at fixed location
space_inds_obs = round(linspace(0, n, total_space_obs, endpoint=False)).astype(
    int
)  # observation locations in the space
time_inds_obs = round(linspace(0, nt, total_time_obs, endpoint=False)).astype(
    int
)  # observation locations along the time
m = total_space_obs * total_time_obs
obs = obs_operator(sigmaR, space_inds_obs, n, time_inds_obs, nt, model)
R = Rmatrix(sigmaR)

# Background class initialization
sigmaB = 0.8  # background error std
# B = Bmatrix(sigmaB,'diagonal')
B = Bmatrix(sigmaB, "diffusion", D=5, M=4)

# Minimization initialization
max_outer = 10  # number of maximum outer loops
max_inner = 500  # number of maximum inner loops
tol = 1e-6  # tolerance for the inner loop
tol_grad = 1e-6  # tolerance for the outer loop
In = eye(n)
F = Precond(B)  # Define the preconditioner

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (2) Generate the truth
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xt = 3.0 * ones(n) + randn(n)  # the true state
xt = model.traj(xt, 5000)  # spin-up
# xt = Bmatrix(sigmaB,'diffusion', D=5, M=4).dot(xt)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (3) Generate the background
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xb = xt + B.sqrtdot(randn(n))
# The square root of B can be used to create correlated errors
# xb = xt + B.sqrtdot(randn(n))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (4) Generate the observations
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = obs.generate_obs(xt)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (5) Variational data assimilation
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter_outer = 0
dxs = []
xa = copy(xb)  # Choose the initial vector
quadcost = math.log10(quadratic_funcval(xa, zeros_like(xa)))
print("")
print("iter", "  CGiter", "        f(x)", "          ||grad(x)||")
iter = 0
while iter_outer < max_outer:  # Gauss-Newton loop
    A = Hessian4dVar(obs, R, B, xa)  # Define the Hessian matrix (Binv + HtRinvH)
    d = obs.misfit(y, xa)  # misfit calculation (y - G(xa))
    b = B.invdot(xb - xa) + obs.adj_gop(xa, R.invdot(d))
    print(
        "{:<9d}{:<9d}{:<20.2f}{:<9.2f}".format(
            iter_outer, iter, nonlinear_funcval(xa), norm(b)
        )
    )
    if norm(b) < tol_grad:
        break
    # Calculate the increment dx such that
    # (Binv + HtRinvH) dx = Binv(xb - x) + Ht Rinv d
    dxs, error, iter, flag = pcg(A, zeros_like(xa), b, F, max_inner, tol)
    dx = dxs[-n:]
    for i in range(iter):
        qval = math.log10(quadratic_funcval(xa, dxs[i * n : (i + 1) * n]))
        quadcost = append(quadcost, qval)
    xa += dx
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
ax.plot(space_inds_obs, y[: len(space_inds_obs)], "og", label="Observations")
ax.plot(xrange, xa, "-r", label="Analysis")
leg = ax.legend()
plt.xlabel("x-coordinate")
plt.ylabel("Temperature")
# plt.show()

# plt.figure()
# plt.plot(quadcost,'r*')
# plt.show()
