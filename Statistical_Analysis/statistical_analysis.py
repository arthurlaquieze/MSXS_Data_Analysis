# Statistical Analysis
# Programming: Selime Gurol (CERFACS), 2021

from numpy import zeros, eye, exp
from numpy.linalg import \
(inv,# To invert a matrix
norm,# To compute the Euclidean norm
cholesky) # To compute Cholesky factorization
from numpy.random import randn # To generate samples from a normalized Gaussian
import matplotlib.pyplot as plt # To plot a grapH


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (1) Initialization
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n  = 4  # state space dimension 

# Observation operator
I = eye(n)
inds_obs = [3] # Location of the observations (1 dimensional grid)
H = I[inds_obs] # H as a selection operator
m = len(inds_obs) # number of observations

# Observation errors
# R is a diagonal matrix
sigmaR = 0.4 # observation error std
R = zeros((m,m))
for ii in range(m):
    R[ii,ii] = sigmaR*sigmaR 

# Background errors
sigmaB = 0.8 # background error std
L = 1.0 # correlation length scale
btype = 'diagonal'
B = zeros((n,n))
if btype == 'diagonal':
    # no correlation between the grid points
    for ii in range(n):
        B[ii,ii] = sigmaB*sigmaB  
if btype == 'soar':
    # correlation between the grid points
    for ii in range(n):
        for jj in range(n):
            rij = abs(jj-ii)
            rho = (1 + rij/L)*exp(-rij/L)
            B[ii,jj] = sigmaB*sigmaB*rho          
# B = B12 * B12^T
B12 = cholesky(B)
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (2) Generate the truth
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xt = randn(n) # the true state

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (3) Generate the background
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xb = # TO DO (use B12)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (4) Generate the observations
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y = # TO DO

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (5) Obtain analysis from BLUE
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Kalman Gain matrix
K = # TO DO
#BLUE analysis
xa = # TO DO

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (5) Diagnostics
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print('')
print('||xt - xb||_2 / ||xt||_2 = ', norm(xt - xb)/norm(xt))
print('||xt - xa||_2 / ||xt||_2 = ', norm(xt - xa)/norm(xt))
print('\n')

# Analysis covariance matrix
Sinv = inv(B) + (H.T).dot(inv(R).dot(H))
S = inv(Sinv)
print('Analysis covariance matrix: \n')
for ii in range(n):
    print('S[{}, {}]: {}'.format(ii, ii, S[ii,ii]))

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                          (6) Plots
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xrange = range(0,n)
fig, ax = plt.subplots()
ax.plot(xrange,xt, '+k', label='Truth')
ax.plot(xrange, xb, 'db', label='Background')
ax.plot(inds_obs, y, 'og', label='Observations')
ax.plot(xrange, xa, 'xr', label='Analysis')
leg = ax.legend()
plt.xlabel('x-coordinate')
plt.ylabel('Temperature')
plt.show()








