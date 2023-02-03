# Class for operators 
# Programming S. Gurol (CERFACS), 2021
#  Include Diffusion operator for B ( O. Goux (CERFACS), 2021 )

from numpy.core.numeric import zeros, zeros_like
from numpy import eye, copy, shape, array, ndarray, ones, sqrt, pi
from numpy.random import randn#, default_rng
from numpy.linalg import norm
from numpy import fft
from scipy import linalg, fft as sp_fft
from scipy import sparse as scp
from scipy.special import gamma
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt # To plot a graph

from solvers import pcg

from scipy import signal

class obs_operator:
    def __init__(self,sigmaR,space_inds,n, time_inds=[], nt=[], model=[]):
        self.sigmaR = sigmaR
        self.space_inds = space_inds
        self.time_inds = time_inds
        self.n = n
        self.nt = nt
        self.model = model

    def generate_obs(self, xt):
        x = copy(xt)
        y = []
        xrange = range(len(x))
        counter = 0
        verbose = 0
        for ii in range(self.nt):
            if ii in self.time_inds:
                new_obs = self.hop(x) + self.sigmaR*randn(len(self.space_inds))
                y.append(new_obs) 
                counter += 1
                if verbose:
                    plt.figure()
                    plt.plot(xrange, x,'-b')
                    plt.plot(self.space_inds, new_obs, '+g')
                    plt.show()
            if counter == len(self.time_inds):
                break
            x = self.model.traj(x,1)
        return array(y).flatten()     

    def misfit(self, y, xt):
        x = copy(xt)
        l = len(self.time_inds)
        nx = len(self.space_inds)
        yy = copy(y.reshape((l,nx)))
        d = []
        counter = 0
        for ii in range(self.nt):
            if ii in self.time_inds:
                d.append(yy[counter] - self.hop(x))
                counter += 1
            if counter == len(self.time_inds):
                break
            x = self.model.traj(x,1)   
        return array(d).flatten()   

    def hop(self, x):
        # Observation operator as a selection operator
        hx = x[self.space_inds]
        return hx

    def tlm_hop(self,dx):
        # Tangent linear model for observation operator
        dy = dx[self.space_inds]
        return dy

    def adj_hop(self,ay):
        # Adjoint model for observation operator
        ax = zeros(self.n) if len(ay.shape) == 1 else zeros((self.n, ay.shape[1]))
        ax[self.space_inds] = ay
        return ax 

    def gop(self, xt):
        # Used in 4DVAR (includes model)
        # Generalized observation operator (H(M(x)))
        x = copy(xt)
        gx = []
        counter = 0
        for ii in range(self.nt):
            if ii in self.time_inds:
                gx.append(self.hop(x))
                counter += 1
            if counter == len(self.time_inds):
                break
            x = self.model.traj(x,1)
        return array(gx).flatten()

    def tlm_gop(self,xt, dxt):
        # Used in 4DVAR (includes model)
        # Tangent linear model for gop : (H(M(x)))
        x = copy(xt)
        dx = copy(dxt)
        dgx = []
        counter = 0
        for ii in range(self.nt):
            if ii in self.time_inds:
                dgx.append(self.tlm_hop(dx))
                counter += 1
            if counter == len(self.time_inds):
                break
            dx = self.model.tlm_traj(x,dx,1)   
            x = self.model.traj(x,1) 
        return array(dgx).flatten()

    def adj_gop(self,xt, axt):
        # Used in 4DVAR (includes model)
        # Adjoint model for gop : (H(M(x)))
        x = copy(xt)
        l = len(self.time_inds)
        nx = len(self.space_inds)
        agx = copy(axt.reshape((l,nx)))
        ax = zeros((self.time_inds[-1]+1,len(xt)))
        counter = 0
        traj_xx = []

        # Forward run
        for ii in range(self.nt):
            traj_xx.append(x)
            x = self.model.traj(x,1)
        for ii in range(self.time_inds[-1],-1,-1):
            if ii in self.time_inds[::-1]:
                ax[ii] += self.adj_hop(agx[l-counter-1])
                counter += 1
            if counter == len(self.time_inds):
                break
            ax[ii-1] = self.model.ad_traj(traj_xx[ii-1],ax[ii],1)
        
        return array(ax[0]).flatten()
            
class Rmatrix:
    def __init__(self, sigmaR):
        self.sigmaR = sigmaR
    def invdot(self,d):
        y = d/(self.sigmaR*self.sigmaR) 
        return y

class Bmatrix:
    def __init__(self, sigmaB, type, D = 10, M=4):
        self.sigmaB2 = sigmaB * sigmaB
        self.sigmaB = sigmaB
        self.sigmaB = sigmaB
        self.D = D
        self.M = M
        self.type = type
        if type == 'diffusion' and D == 0:
            self.type = 'diagonal'
    def invdot(self,x):
        if norm(x) == 0.:
            return zeros_like(x)
        if self.type == 'diagonal':
            y = x/(self.sigmaB2) 
            return y
        if self.type == 'diffusion':
            n = len(x)
            h = 1 # grid resolution
            M = self.M
            l = self.D/sqrt(2*M-3)
            #Initialize finite differences matrix T = I -2*(l/h)**2 * Laplacian
            T = scp.diags(
                [(1 + 2*(l/h)**2)*ones(n), -(l/h)**2*ones(n), -(l/h)**2*ones(n)],
                           [0,-1,1], format = 'csr')
            T += scp.csr_matrix(([-(l/h)**2, -(l/h)**2], ([0, n-1], [0, n-1]) ))
            #Normalization factor for the infinite line
            normalization = sqrt( 2 * gamma(M) * sqrt(pi) * l / gamma(M-0.5) )
            #Apply operator
            y = x / self.sigmaB
            y /= normalization
            y *= sqrt(h)
            for k in range(M):
                y = T.dot(y)
            y *= sqrt(h)
            y /= normalization
            y /= self.sigmaB
            return y
    def dot(self,x):
        if norm(x) == 0.:
            return zeros_like(x)
        if self.type == 'diagonal':
            y = self.sigmaB2 * x
            return y
        if self.type == 'diffusion':
            n = len(x)
            h = 1 # grid resolution
            M = self.M
            l = self.D/sqrt(2*M-3)
            #Initialize finite differences matrix T = I -2*(l/h)**2 * Laplacian
            T = scp.diags(
                [(1 + 2*(l/h)**2)*ones(n), -(l/h)**2*ones(n), -(l/h)**2*ones(n)],
                           [0,-1,1], format = 'csr')
            T += scp.csr_matrix(([-(l/h)**2, -(l/h)**2], ([0, n-1], [0, n-1]) ))
            #Normalization factor for the infinite line
            normalization = sqrt( 2 * gamma(M) * sqrt(pi) * l / gamma(M-0.5) )
            #Apply operator
            y = x * self.sigmaB
            y *= normalization
            y /= sqrt(h)#
            for k in range(M):
                y = scp.linalg.spsolve(T,y)
            y /= sqrt(h)
            y *= normalization
            y *= self.sigmaB
            return y
    def sqrtdot(self,x):
        if norm(x) == 0.:
            return zeros_like(x)
        if self.type == 'diagonal':
            y = self.sigmaB * x
            return y
        if self.type == 'diffusion':
            n = len(x)
            h = 1 # grid resolution
            M = self.M
            l = self.D/sqrt(2*M-3)
            #Initialize finite differences matrix T = I -2*(l/h)**2 * Laplacian
            T = scp.diags(
                [(1 + 2*(l/h)**2)*ones(n), -(l/h)**2*ones(n), -(l/h)**2*ones(n)],
                           [0,-1,1], format = 'csr')
            T += scp.csr_matrix(([-(l/h)**2, -(l/h)**2], ([0, n-1], [0, n-1]) ))
            #Normalization factor for the infinite line
            normalization = sqrt( 2 * gamma(M) * sqrt(pi) * l / gamma(M-0.5) )
            #Apply operator
            y = x * self.sigmaB
            y *= normalization
            y /= sqrt(h)#
            for k in range(M//2):
                y = scp.linalg.spsolve(T,y)
            return y
        
class Hessian3dVar:
    def __init__(self,obs,R,B):
        self.obs = obs
        self.R = R
        self.B = B
    def dot(self,dx):
        # return dy = (Binv + HtRinvH) *dx
        return dy

class Hessian4dVar:
    def __init__(self,obs,R,B,xt):
        self.obs = obs
        self.R = R
        self.B = B
        self.xt = copy(xt)
    def dot(self,dx):
        w = self.R.invdot(self.obs.tlm_gop(self.xt,dx))
        gtrinvg_dx = self.obs.adj_gop(self.xt,w)
        binv_dx = self.B.invdot(dx)
        dy =  binv_dx +  gtrinvg_dx
        return dy         

class Precond:
    def __init__(self, F):
        self.F = F

    def dot(self,x):
        y = (self.F).dot(x)
        return y 





    

    

