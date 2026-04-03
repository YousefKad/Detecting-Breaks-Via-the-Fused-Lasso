"""
Monte Carlo experiment for Kaddoura and Westerlund (2023),
"Estimation of Panel Data Models with Random Interactive Effects and Multiple
Structural Breaks when T is Fixed" (Journal of Business & Economic Statistics).

This script reproduces the fused-lasso Monte Carlo design used to study
break detection in panel data models with fixed T and large N.

Core configuration parameters
-----------------------------
p : int
    Number of slope regressors before adding the lagged dependent variable.
n : int
    Number of cross-sectional units.
T : int
    Number of time periods in the raw DGP. For DATA3, the effective time
    dimension becomes T-1 after constructing the lagged dependent variable.
m : int
    Number of true breaks in the coefficient path.
r : int
    Number of latent common factors.
phi : float
    Persistence parameter in the factor process F_t.
phi_1 : float
    Persistence / spatial spillover parameter in the idiosyncratic disturbance.
pi : float
    Persistence / spatial spillover parameter in the regressor innovation.
lambd_values : ndarray
    Candidate tuning parameters used by the information criterion.
sim : int
    Number of Monte Carlo repetitions.

Notes
-----
The local package ``TimeSeriesP`` provides the lag operator through

    from TimeSeriesP.lag import Lag

so the repository can be run from the project root without editing import paths.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cvxpy as cp
from tabulate import tabulate
from TimeSeriesP.lag import Lag

"""
Description: Same as code with less for-loops. mainly in generating y and u (so far). Check the differences.
"""

## Generating data:



class DATA1:
    def __init__(self, m, T,n,p):
        self.m = m
        self.T = T
        self.p = p
        self.n = n
    def DGP1(self):
        """
       1 BREAK
        """
        #np.random.seed(1)
        beta   = np.zeros((self.p,self.T))
        beta_b = {}
        for i in range(self.m+1):
            beta_b["beta_" + str(i)] = np.random.normal(0,1,self.p)
        for t in range(self.T):
            if t < math.floor(self.T/(self.m+1)):
                beta[0:self.p,t] =  beta_b["beta_" + str(0)]
            else:
                 beta[0:self.p,t] =  beta_b["beta_" + str(1)]
        eps  = np.random.normal(0,1,(self.T,self.n))
        X    = np.random.normal(0,1,(self.T,self.n,self.p))
        y    = np.zeros((self.T,self.n))
        for t in range(self.T):
            y[t] = X[t]@beta[0:self.p,t] + eps[t]     
        return X,y,beta,eps
    def DGP2(self):
        """
        2 BREAKS
        """
        #np.random.seed(1)
        beta   = np.zeros((self.p,self.T))
        beta_b = {}
        for i in range(self.m+1):
            beta_b["beta_" + str(i)] = np.random.normal(0,1,self.p)
        for t in range(self.T):
            if t < math.floor(self.T/(self.m+1)):
                beta[0:self.p,t] =  beta_b["beta_" + str(0)]
            elif math.floor(self.T/(self.m+1))<=t<math.floor(2*(self.T/(self.m+1))):
                beta[0:self.p,t] =  beta_b["beta_" + str(1)]
            else:
                 beta[0:self.p,t] =  beta_b["beta_" + str(2)]
        eps  = np.random.normal(0,1,(self.T,self.n))
        X    = np.random.normal(0,1,(self.T,self.n,self.p))
        y    = np.zeros((self.T,self.n))
        for t in range(self.T):
            y[t] = X[t]@beta[0:self.p,t] + eps[t]      
        return X,y,beta,eps
    
    def DGPA(self):
        """
        all breaks
        """
        #np.random.seed(1)
        beta   = np.random.normal(0,1,(self.p,self.T))
        eps  = np.random.normal(0,1,(self.T,self.n))
        X    = np.random.normal(0,1,(self.T,self.n,self.p))
        y    = np.zeros((self.T,self.n))
        for t in range(self.T):
            y[t] = X[t]@beta[0:self.p,t] + eps[t] 
        return X,y,beta,eps
    def DGPO(self):
        """
        No breaks
        """
        #np.random.seed(1)
        beta   = np.zeros((self.p,self.T))
        beta_b = {}
        for i in range(self.m+1):
            beta_b["beta_" + str(i)] = np.random.normal(0,1,self.p)
        for t in range(self.T):
            beta[0:self.p,t] = beta_b["beta_" + str(0)]
        eps  = np.random.normal(0,1,(self.T,self.n))
        X    = np.random.normal(0,1,(self.T,self.n,self.p))
        y    = np.zeros((self.T,self.n))
        for t in range(self.T):
            y[t] = X[t]@beta[0:self.p,t] + eps[t]       
        return X,y,beta,eps




class DATA2:
    
    def __init__(self,r, m, T,n,p):
        self.m = m
        self.T = T
        self.p = p
        self.n = n
        self.r = r
    def DGP1(self):
        
        """
        1 BREAK
        """
        #np.random.seed(1)
        beta     = np.zeros((self.p,self.T))
        beta_b   = {}
        for i in range(self.m+1):
            beta_b["beta_" + str(i)] = i*np.ones(self.p) #np.random.normal(0,1,self.p)
        for t in range(self.T):
            if t < math.floor(self.T/(self.m+1)):
                beta[0:self.p,t] =  beta_b["beta_" + str(0)]
            else:
                beta[0:self.p,t] =  beta_b["beta_" + str(1)]
        lambd   = np.random.normal(1,1,(self.n,self.r)) 
        F       = np.random.normal(0,1,(self.T,self.r))
        eps     = np.random.normal(0,1,(self.T,self.n))
        u       = F@lambd.T + eps
        X       = np.random.normal(0,1,(self.T,self.n,self.p))
        u_cmean = np.zeros((self.T,self.n))
        for t in range(self.T):
            u_cmean[t] = np.mean(u[t])
        u_tilde = u - u_cmean
        X_mean  = np.zeros((self.T,self.p))
        for t in range(self.T):
            for i in range(self.p):
                X_mean[t,i] = np.mean(X[t][0:self.n,i])
        X_tilde = np.zeros(((self.T,self.n,self.p)))
        for t in range(self.T):
            for i in range(self.p):
                X_tilde[t][0:self.n,i] = X[t][0:self.n,i]- X_mean[t,i]
        y = np.zeros((self.T,self.n))
        for t in range(self.T):
            y[t] = X[t]@beta[0:self.p,t] + u[t] 
        y_tilde = np.zeros((self.T,self.n))
        for t in range(self.T):
            y_tilde[t] = X_tilde[t]@beta[0:self.p,t] + u_tilde[t] 
        return X,y,beta,u,eps,lambd,F,y_tilde,u_tilde,X_mean,X_tilde
    
    def DGP2(self):
        
        """
        2 BREAK
        """
        #np.random.seed(1)
        beta     = np.zeros((self.p,self.T))
        beta_b   = {}
        for i in range(self.m+1):
            beta_b["beta_" + str(i)] =  i*np.ones(self.p) #np.random.normal(0,1,self.p)
        for t in range(self.T):
            if t < math.floor(self.T/(self.m+1)):
                beta[0:self.p,t] =  beta_b["beta_" + str(0)]
            elif math.floor(self.T/(self.m+1))<=t<math.floor(2*(self.T/(self.m+1))):
                beta[0:self.p,t] =  beta_b["beta_" + str(1)]
            else:
                 beta[0:self.p,t] =  beta_b["beta_" + str(2)]
        lambd   = np.random.normal(1,1,(self.n,self.r)) 
        F       = np.random.normal(0,1,(self.T,self.r))
        eps     = np.random.normal(0,1,(self.T,self.n))
        u       = F@lambd.T + eps
        X       = np.random.normal(0,1,(self.T,self.n,self.p))
        u_cmean = np.zeros((self.T,self.n))
        for t in range(self.T):
            u_cmean[t] = np.mean(u[t])
        u_tilde = u - u_cmean
        X_mean  = np.zeros((self.T,self.p))
        for t in range(self.T):
            for i in range(self.p):
                X_mean[t,i] = np.mean(X[t][0:self.n,i])
        X_tilde = np.zeros(((self.T,self.n,self.p)))
        for t in range(self.T):
            for i in range(self.p):
                X_tilde[t][0:self.n,i] = X[t][0:self.n,i]- X_mean[t,i]
        y = np.zeros((self.T,self.n))
        for t in range(self.T):
            y[t] = X[t]@beta[0:self.p,t] + u[t] 
        y_tilde = np.zeros((self.T,self.n))
        for t in range(self.T):
            y_tilde[t] = X_tilde[t]@beta[0:self.p,t] + u_tilde[t] 
        return X,y,beta,u,eps,lambd,F,y_tilde,u_tilde,X_mean,X_tilde
    
    def DGPA(self):
        
        """
        breaks for all t
        """
        #np.random.seed(1)
        #beta   = np.random.normal(0,1,(self.p,self.T))
        beta     = np.zeros((self.p,self.T))
        beta_b   = {}
        for i in range(self.m+1):
            beta_b["beta_" + str(i)] = i*np.ones(self.p) #np.random.normal(0,1,self.p)
        for t in range(self.T):
            beta[0:self.p,t] =  beta_b["beta_" + str(t)]
        lambd   = np.random.normal(1,1,(self.n,self.r)) 
        F       = np.random.normal(0,1,(self.T,self.r))
        eps     = np.random.normal(0,1,(self.T,self.n))
        u       = F@lambd.T + eps
        X       = np.random.normal(0,1,(self.T,self.n,self.p))
        u_cmean = np.zeros((self.T,self.n))
        for t in range(self.T):
            u_cmean[t] = np.mean(u[t])
        u_tilde = u - u_cmean
        X_mean  = np.zeros((self.T,self.p))
        for t in range(self.T):
            for i in range(self.p):
                X_mean[t,i] = np.mean(X[t][0:self.n,i])
        X_tilde = np.zeros(((self.T,self.n,self.p)))
        for t in range(self.T):
            for i in range(self.p):
                X_tilde[t][0:self.n,i] = X[t][0:self.n,i]- X_mean[t,i]
        y = np.zeros((self.T,self.n))
        for t in range(self.T):
            y[t] = X[t]@beta[0:self.p,t] + u[t] 
        y_tilde = np.zeros((self.T,self.n))
        for t in range(self.T):
            y_tilde[t] = X_tilde[t]@beta[0:self.p,t] + u_tilde[t] 
        return X,y,beta,u,eps,lambd,F,y_tilde,u_tilde,X_mean,X_tilde
    
    def DGPO(self):
        
        """
        no breaks
        """
        #np.random.seed(1)
        beta     = 0*np.ones((self.p,self.T))
        lambd   = np.random.normal(0,1,(self.n,self.r)) 
        F       = np.random.normal(0,1,(self.T,self.r))
        eps     = np.random.normal(0,1,(self.T,self.n))
        u       = F@lambd.T + eps
        lambd   = np.random.normal(1,1,(self.n,self.r)) 
        F       = np.random.normal(0,1,(self.T,self.r))
        eps     = np.random.normal(0,1,(self.T,self.n))
        u       = F@lambd.T + eps
        X       = np.random.normal(0,1,(self.T,self.n,self.p))
        u_cmean = np.zeros((self.T,self.n))
        for t in range(self.T):
            u_cmean[t] = np.mean(u[t])
        u_tilde = u - u_cmean
        X_mean  = np.zeros((self.T,self.p))
        for t in range(self.T):
            for i in range(self.p):
                X_mean[t,i] = np.mean(X[t][0:self.n,i])
        X_tilde = np.zeros(((self.T,self.n,self.p)))
        for t in range(self.T):
            for i in range(self.p):
                X_tilde[t][0:self.n,i] = X[t][0:self.n,i]- X_mean[t,i]
        y = np.zeros((self.T,self.n))
        for t in range(self.T):
            y[t] = X[t]@beta[0:self.p,t] + u[t] 
        y_tilde = np.zeros((self.T,self.n))
        for t in range(self.T):
            y_tilde[t] = X_tilde[t]@beta[0:self.p,t] + u_tilde[t] 
        return X,y,beta,u,eps,lambd,F,y_tilde,u_tilde,X_mean,X_tilde
    
    
    
    
class DATA3:
    
    def __init__(self,r, m, T,n,p,phi,phi_1,pi):
        self.r = r
        self.m = m
        self.T = T
        self.n = n
        self.p = p
        self.phi = phi
        self.phi_1 = phi_1
        self.pi    = pi
        
    def DGP1(self):
         
        """
        1 BREAK
        """
        #np.random.seed(1)
        beta     = np.zeros((self.p,self.T))
        beta_b   = {}
        rho      = np.ones((self.T))
        for i in range(self.m+1):
            beta_b["beta_" + str(i)] = i*np.ones(self.p) #np.random.normal(0,1,self.p)
        for t in range(self.T):
            if t < math.floor(self.T/(self.m+1)):
                beta[0:self.p,t] =  beta_b["beta_" + str(0)]
                rho[t]           = 0.5
            else:
                beta[0:self.p,t] =  beta_b["beta_" + str(1)]
                rho[t]           = 1
        lambd_y   = np.random.normal(2,1,(self.n,self.r)) 
        F         = np.zeros((self.T,self.r))
        f_0       = np.zeros((1,self.r))
        eta       = np.random.normal(0,1,(self.T,self.r))
        for t in range(self.T):
            if t==0:
                F[t] = (1-self.phi) + self.phi*f_0 + eta[t]
            else:
                F[t] = (1-self.phi) + self.phi*F[t-1] + eta[t]
        sigma_i = np.random.uniform(0.5,1,(self.n,1))
        xi  = np.zeros((self.T,self.n))
        for i in range(self.n):
            for t in range(self.T):
                xi[t,i] = np.random.normal(0,sigma_i[i])
        xi_rev = xi[::,::-1]
        eps   = np.zeros((self.T,self.n))
        for t in range(self.T):
            for i in range(self.n):
                if t==0:
                    eps[t,i] = xi[t,i] + np.sum(self.phi_1*(xi[t,i+1:i+11]))+ np.sum(self.phi_1*xi_rev[t,self.n-i:self.n+10-i])
                else:
                    eps[t,i] = self.phi_1*eps[t-1,i] + xi[t,i] + np.sum(self.phi_1*(xi[t,i+1:i+11]))+ np.sum(self.phi_1*xi_rev[t,self.n-i:self.n+10-i])
        u       =  eps
        lambd_x   = np.random.normal(2,1,(self.n,self.p,self.r)) 
        X         = np.zeros((self.T,self.n,self.p))
        e         = np.random.normal(0,1,(self.T,self.n,self.p))
        e_rev     = e[::,::-1,::]
        nu  = np.zeros((self.T,self.n,self.p))
        for i in range(self.n):
            for t in range(self.T):
                for k in range(self.p):
                    if t==0:
                        nu[t,i,k] = e[t,i,k] + np.sum(self.pi*(e[t,i+1:i+11,k]))+ np.sum(self.pi*e_rev[t,self.n-i:self.n+10-i,k])
                    else:
                        nu[t,i,k] = self.pi*nu[t-1,i,k]+ e[t,i,k] + np.sum(self.pi*(e[t,i+1:i+11,k]))+ np.sum(self.pi*e_rev[t,self.n-i:self.n+10-i,k])
        for t in range(self.T):
            for i in range(self.n):
                for k in range(self.p):
                    X[t,i,k] = F[t]@lambd_x[i,k] + nu[t,i,k]
        u_cmean = np.zeros((self.T,self.n))
        for t in range(self.T):
            u_cmean[t] = np.mean(u[t])
        u_tilde = u - u_cmean
        y   = np.zeros((self.T,self.n))
        FE  = np.random.normal(1,1,(self.T,)) 
        z = np.zeros((self.T,self.n))
        for t in range(self.T):
            z[t] = X[t]@beta[0:self.p,t] + u[t] 
        for t in range(self.T):
            for i in range(self.n):
                if t == 0:
                    y[t,i] = z[t,i]
                else:
                    y[t,i] = FE[t] + rho[t]*y[t-1,i] + z[t,i]
        y_lag = Lag(self.T)@y
        y_lag = y_lag[1:self.T,]
        y     = y[1:self.T,]
        X     = X[1:self.T,]
        u     = u[1:self.T,]
        y_lag = y_lag.reshape((self.T-1)*self.n,1)
        X = X.reshape((self.T-1)*self.n,self.p)
        X = np.c_[y_lag, X]
        rho = rho[1:self.T]
        beta = beta[:,1:self.T]
        self.p = self.p + 1
        self.T = self.T - 1
        X = X.reshape((self.T,self.n,self.p))
        X_mean  = np.zeros((self.T,self.p))
        for t in range(self.T):
            for i in range(self.p):
                X_mean[t,i] = np.mean(X[t][0:self.n,i])
        X_tilde = np.zeros(((self.T,self.n,self.p)))
        for t in range(self.T):
            for i in range(self.p):
                X_tilde[t][0:self.n,i] = X[t][0:self.n,i]- X_mean[t,i]
        y_tilde = np.zeros((self.T,self.n))
        y_cmeant = np.zeros((self.T,self.n))
        for t in range(self.T):
            y_cmeant[t] = np.mean(y[t])
        y_tilde   = y -  y_cmeant 
        beta = np.concatenate((rho.reshape((1,self.T)),beta),axis=0)
        return X,y,beta,u,eps,F,y_tilde,u_tilde,X_mean,X_tilde
                        
    def DGP2(self):
         
        """
        2 BREAK
        """
       #np.random.seed(1)
        beta     = np.zeros((self.p,self.T))
        beta_b   = {}
        rho      = np.ones((self.T))
        for i in range(self.m+1):
            beta_b["beta_" + str(i)] =  i*np.ones(self.p) #np.random.normal(0,1,self.p)
        for t in range(self.T):
            if t < math.floor(self.T/(self.m+1)) + 1:
                beta[0:self.p,t] =  beta_b["beta_" + str(0)]
                rho[t]           = 0.5
            elif math.floor(self.T/(self.m+1))<=t<math.floor(2*(self.T/(self.m+1))):
                beta[0:self.p,t] =  beta_b["beta_" + str(1)]
                rho[t]           = 0.9
            else:
                beta[0:self.p,t] =  beta_b["beta_" + str(2)]
                rho[t]           = 1
        lambd_y   = np.random.normal(2,1,(self.n,self.r)) 
        F         = np.zeros((self.T,self.r))
        f_0       = np.zeros((1,self.r))
        eta       = np.random.normal(0,1,(self.T,self.r))
        for t in range(self.T):
            if t==0:
                F[t] = (1-self.phi) + self.phi*f_0 + eta[t]
            else:
                F[t] = (1-self.phi) + self.phi*F[t-1] + eta[t]
        sigma_i = np.random.uniform(0.5,1,(self.n,1))
        xi  = np.zeros((self.T,self.n))
        for i in range(self.n):
            for t in range(self.T):
                xi[t,i] = np.random.normal(0,sigma_i[i])
        xi_rev = xi[::,::-1]
        eps   = np.zeros((self.T,self.n))
        for t in range(self.T):
            for i in range(self.n):
                if t==0:
                    eps[t,i] = xi[t,i] + np.sum(self.phi_1*(xi[t,i+1:i+11]))+ np.sum(self.phi_1*xi_rev[t,self.n-i:self.n+10-i])
                else:
                    eps[t,i] = self.phi_1*eps[t-1,i] + xi[t,i] + np.sum(self.phi_1*(xi[t,i+1:i+11]))+ np.sum(self.phi_1*xi_rev[t,self.n-i:self.n+10-i])
        u       =  eps
        lambd_x   = np.random.normal(2,1,(self.n,self.p,self.r)) 
        X         = np.zeros((self.T,self.n,self.p))
        e         = np.random.normal(0,1,(self.T,self.n,self.p))
        e_rev     = e[::,::-1,::]
        nu  = np.zeros((self.T,self.n,self.p))
        for i in range(self.n):
            for t in range(self.T):
                for k in range(self.p):
                    if t==0:
                        nu[t,i,k] = e[t,i,k] + np.sum(self.pi*(e[t,i+1:i+11,k]))+ np.sum(self.pi*e_rev[t,self.n-i:self.n+10-i,k])
                    else:
                        nu[t,i,k] = self.pi*nu[t-1,i,k]+ e[t,i,k] + np.sum(self.pi*(e[t,i+1:i+11,k]))+ np.sum(self.pi*e_rev[t,self.n-i:self.n+10-i,k])
        for t in range(self.T):
            for i in range(self.n):
                for k in range(self.p):
                    X[t,i,k] = F[t]@lambd_x[i,k] + nu[t,i,k]
        u_cmean = np.zeros((self.T,self.n))
        for t in range(self.T):
            u_cmean[t] = np.mean(u[t])
        u_tilde = u - u_cmean
        y   = np.zeros((self.T,self.n))
        FE  = np.random.normal(0,1,(self.T,)) 
        z = np.zeros((self.T,self.n))
        for t in range(self.T):
            z[t] = X[t]@beta[0:self.p,t] + u[t] 
        for t in range(self.T):
            for i in range(self.n):
                if t == 0:
                    y[t,i] = z[t,i]
                else:
                    y[t,i] = FE[t] + rho[t]*y[t-1,i] + z[t,i]
        y_lag = Lag(self.T)@y
        y_lag = y_lag[1:self.T,]
        y     = y[1:self.T,]
        X     = X[1:self.T,]
        u     = u[1:self.T,]
        y_lag = y_lag.reshape((self.T-1)*self.n,1)
        X = X.reshape((self.T-1)*self.n,self.p)
        X = np.c_[y_lag, X]
        rho = rho[1:self.T]
        beta = beta[:,1:self.T]
        self.p = self.p + 1
        self.T = self.T - 1
        X = X.reshape((self.T,self.n,self.p))
        X_mean  = np.zeros((self.T,self.p))
        for t in range(self.T):
            for i in range(self.p):
                X_mean[t,i] = np.mean(X[t][0:self.n,i])
        X_tilde = np.zeros(((self.T,self.n,self.p)))
        for t in range(self.T):
            for i in range(self.p):
                X_tilde[t][0:self.n,i] = X[t][0:self.n,i]- X_mean[t,i]
        y_tilde = np.zeros((self.T,self.n))
        y_cmeant = np.zeros((self.T,self.n))
        for t in range(self.T):
            y_cmeant[t] = np.mean(y[t])
        y_tilde   = y -  y_cmeant 
        beta = np.concatenate((rho.reshape((1,self.T)),beta),axis=0)
        return X,y,beta,u,eps,F,y_tilde,u_tilde,X_mean,X_tilde
    
    
    def DGPA(self):

        """
        breaks for all t
        """
        #np.random.seed(1)
        #beta   = np.random.normal(0,1,(self.p,self.T))
        beta     = np.zeros((self.p,self.T))
        beta_b   = {}
        for i in range(self.m+1):
            beta_b["beta_" + str(i)] = i*np.ones(self.p) #np.random.normal(0,1,self.p)
        for t in range(self.T):
            beta[0:self.p,t] =  beta_b["beta_" + str(t)]
        lambd_y   =   np.random.normal(2,1,(self.n,self.r)) 
        F         = np.zeros((self.T,self.r))
        f_0       = np.zeros((1,self.r))
        eta       = np.random.normal(0,1,(self.T,self.r))
        for t in range(self.T):
            if t==0:
                F[t] = (1-self.phi) + self.phi*f_0 + eta[t]
            else:
                F[t] = (1-self.phi) + self.phi*F[t-1] + eta[t]
        sigma_i = np.random.uniform(0.5,1,(self.n,1))
        xi  = np.zeros((self.T,self.n))
        for i in range(self.n):
            for t in range(self.T):
                xi[t,i] = np.random.normal(0,sigma_i[i])
        xi_rev = xi[::,::-1]
        eps   = np.zeros((self.T,self.n))
        for t in range(self.T):
            for i in range(self.n):
                if t==0:
                    eps[t,i] = xi[t,i] + np.sum(self.phi_1*(xi[t,i+1:i+11]))+ np.sum(self.phi_1*xi_rev[t,self.n-i:self.n+10-i])
                else:
                    eps[t,i] = self.phi_1*eps[t-1,i] + xi[t,i] + np.sum(self.phi_1*(xi[t,i+1:i+11]))+ np.sum(self.phi_1*xi_rev[t,self.n-i:self.n+10-i])
        u       = F@lambd_y.T + eps
        lambd_x   = np.random.normal(2,1,(self.n,self.p,self.r)) 
        X         = np.zeros((self.T,self.n,self.p))
        e         = np.random.normal(0,1,(self.T,self.n,self.p))
        e_rev     = e[::,::-1,::]
        nu  = np.zeros((self.T,self.n,self.p))
        for i in range(self.n):
            for t in range(self.T):
                for k in range(self.p):
                    if t==0:
                        nu[t,i,k] = e[t,i,k] + np.sum(self.pi*(e[t,i+1:i+11,k]))+ np.sum(self.pi*e_rev[t,self.n-i:self.n+10-i,k])
                    else:
                        nu[t,i,k] = self.pi*nu[t-1,i,k]+ e[t,i,k] + np.sum(self.pi*(e[t,i+1:i+11,k]))+ np.sum(self.pi*e_rev[t,self.n-i:self.n+10-i,k])
        for t in range(self.T):
            for i in range(self.n):
                for k in range(self.p):
                    X[t,i,k] = F[t]@lambd_x[i,k] + nu[t,i,k]
        u_cmean = np.zeros((self.T,self.n))
        for t in range(self.T):
            u_cmean[t] = np.mean(u[t])
        u_tilde = u - u_cmean
        X_mean  = np.zeros((self.T,self.p))
        for t in range(self.T):
            for i in range(self.p):
                X_mean[t,i] = np.mean(X[t][0:self.n,i])
        X_tilde = np.zeros(((self.T,self.n,self.p)))
        for t in range(self.T):
            for i in range(self.p):
                X_tilde[t][0:self.n,i] = X[t][0:self.n,i]- X_mean[t,i]
        y = np.zeros((self.T,self.n))
        for t in range(self.T):
            y[t] = X[t]@beta[0:self.p,t] + u[t] 
        y_tilde = np.zeros((self.T,self.n))
        for t in range(self.T):
            y_tilde[t] = X_tilde[t]@beta[0:self.p,t] + u_tilde[t] 
        return X,y,beta,u,eps,F,y_tilde,u_tilde,X_mean,X_tilde
                        
    def DGPO(self):

        """
        no breaks
        """
        #np.random.seed(1)
        beta      = 0*np.ones((self.p,self.T))
        rho       = 1*np.ones((self.T))
        lambd_y   = np.random.normal(2,1,(self.n,self.r)) 
        F         = np.zeros((self.T,self.r))
        f_0       = np.zeros((1,self.r))
        eta       = np.random.normal(0,1,(self.T,self.r))
        for t in range(self.T):
            if t==0:
                F[t] = (1-self.phi) + self.phi*f_0 + eta[t]
            else:
                F[t] = (1-self.phi) + self.phi*F[t-1] + eta[t]
        sigma_i = np.random.uniform(0.5,1,(self.n,1))
        xi  = np.zeros((self.T,self.n))
        for i in range(self.n):
            for t in range(self.T):
                xi[t,i] = np.random.normal(0,sigma_i[i])
        xi_rev = xi[::,::-1]
        eps   = np.zeros((self.T,self.n))
        for t in range(self.T):
            for i in range(self.n):
                if t==0:
                    eps[t,i] = xi[t,i] + np.sum(self.phi_1*(xi[t,i+1:i+11]))+ np.sum(self.phi_1*xi_rev[t,self.n-i:self.n+10-i])
                else:
                    eps[t,i] = self.phi_1*eps[t-1,i] + xi[t,i] + np.sum(self.phi_1*(xi[t,i+1:i+11]))+ np.sum(self.phi_1*xi_rev[t,self.n-i:self.n+10-i])
        u       =  eps
        lambd_x   = np.random.normal(2,1,(self.n,self.p,self.r)) 
        X         = np.zeros((self.T,self.n,self.p))
        e         = np.random.normal(0,1,(self.T,self.n,self.p))
        e_rev     = e[::,::-1,::]
        nu  = np.zeros((self.T,self.n,self.p))
        for i in range(self.n):
            for t in range(self.T):
                for k in range(self.p):
                    if t==0:
                        nu[t,i,k] = e[t,i,k] + np.sum(self.pi*(e[t,i+1:i+11,k]))+ np.sum(self.pi*e_rev[t,self.n-i:self.n+10-i,k])
                    else:
                        nu[t,i,k] = self.pi*nu[t-1,i,k]+ e[t,i,k] + np.sum(self.pi*(e[t,i+1:i+11,k]))+ np.sum(self.pi*e_rev[t,self.n-i:self.n+10-i,k])
        for t in range(self.T):
            for i in range(self.n):
                for k in range(self.p):
                    X[t,i,k] = F[t]@lambd_x[i,k] + nu[t,i,k]
        u_cmean = np.zeros((self.T,self.n))
        for t in range(self.T):
            u_cmean[t] = np.mean(u[t])
        u_tilde = u - u_cmean
        y   = np.zeros((self.T,self.n))
        FE  = np.random.normal(0,1,(self.T,)) 
        z = np.zeros((self.T,self.n))
        for t in range(self.T):
            z[t] = X[t]@beta[0:self.p,t] + u[t] 
        for t in range(self.T):
            for i in range(self.n):
                if t == 0:
                    y[t,i] = z[t,i]
                else:
                    y[t,i] = FE[t] + rho[t]*y[t-1,i] + z[t,i]
        y_lag = Lag(self.T)@y
        y_lag = y_lag[1:self.T,]
        y     = y[1:self.T,]
        X     = X[1:self.T,]
        u     = u[1:self.T,]
        y_lag = y_lag.reshape((self.T-1)*self.n,1)
        X = X.reshape((self.T-1)*self.n,self.p)
        X = np.c_[y_lag, X]
        rho = rho[1:self.T]
        beta = beta[:,1:self.T]
        self.p = self.p + 1
        self.T = self.T - 1
        X = X.reshape((self.T,self.n,self.p))
        X_mean  = np.zeros((self.T,self.p))
        for t in range(self.T):
            for i in range(self.p):
                X_mean[t,i] = np.mean(X[t][0:self.n,i])
        X_tilde = np.zeros(((self.T,self.n,self.p)))
        for t in range(self.T):
            for i in range(self.p):
                X_tilde[t][0:self.n,i] = X[t][0:self.n,i]- X_mean[t,i]
        y_tilde = np.zeros((self.T,self.n))
        y_cmeant = np.zeros((self.T,self.n))
        for t in range(self.T):
            y_cmeant[t] = np.mean(y[t])
        y_tilde   = y -  y_cmeant 
        beta = np.concatenate((rho.reshape((1,self.T)),beta),axis=0)
        return X,y,beta,u,eps,F,y_tilde,u_tilde,X_mean,X_tilde
                        


class Optimize:
     def __init__(self,p,T,n):
         self.p = p
         self.T = T
         self.n = n
     def OLS(self,X,y):
         b_ols = cp.Variable((self.p,self.T))
         Obj_ols = 0
         for t in range(self.T):
             Obj_ols += (1/self.n)*cp.norm((y[t]-X[t]@b_ols[0:self.p,t]),p=2)**2
        
         obj_olsm  = cp.Minimize(Obj_ols)
         prob_olsm = cp.Problem(obj_olsm)
         prob_olsm.solve()
         b_olsm = b_ols.value
         return b_olsm,prob_olsm.status,prob_olsm.value
     def FGLS(self,X,y,b_o,Lambda):
          m_b     = 0
          Obj   = 0
          b     = cp.Variable((self.p,self.T))
          for t in range(self.T):
              if t==0:
                  Obj += (1/self.n)*cp.norm(y[t]-X[t]@b[0:self.p,t],p=2)**2
              else:
                  Obj += (1/self.n)*cp.norm(y[t]-X[t]@b[0:self.p,t],p=2)**2 + (Lambda*cp.pnorm(b[0:self.p,t]- b[0:self.p,t-1],p=2))*(cp.pnorm(b_o[0:self.p,t]-b_o[0:self.p,t-1],p=2)**(-2))
          Obj_m = cp.Minimize(Obj)
          prob  = cp.Problem(Obj_m)
          prob.solve()
          for t in range(self.T):
              if t==0:
                  pass
              elif np.linalg.norm(b.value[0:self.p,t]- b.value[0:self.p,t-1],2) > 0.001:
                  m_b+= 1
              else:
                  m_b +=0
          return  b.value, m_b, prob.status,prob.value
     def NBOLS(self,X,y):
          X_o = X.reshape((self.T*self.n,self.p))
          y_o = y.reshape((self.T*self.n,1))
          b_nb = cp.Variable(self.p)
          objective_nb = (y_o-X_o@b_nb).T@(y_o-X_o@b_nb).T
          objective_nbm = cp.Minimize(objective_nb)
          prob_nb = cp.Problem(objective_nbm)
          prob_nb.solve()
          b_nba = np.linalg.inv((X_o.T@X_o))@X_o.T@y_o
          return  b_nb.value,b_nba , prob_nb.status,prob_nb.value
      
  
def IC(lambd_values,y,X,p,T,n):
    opt = Optimize(p,T,n)
    IC_vector = np.zeros((len(lambd_values),1))
    m_breaks = np.zeros((len(lambd_values),1))
    for l in range(len(lambd_values)):
        try:
            Lambda = lambd_values[l]
            b_w = opt.OLS(X, y)[0] 
            b_l,m_b,_ ,_ = opt.FGLS(X,y,b_w,Lambda)
            IC  = (1*(np.log(n*T))/(np.sqrt(n*T)))*p*(m_b+1) #does quite well for even 0.1. The less the worse though, probably the hiigher the worse. INTERVAL needed.
            for t in range(T):
                IC += (1/(n*T))*(np.linalg.norm((y[t]-X[t]@b_l[0:p,t]),2))**2
            m_breaks[l]  = m_b
            IC_vector[l] =  IC
        except:
            m_breaks[l]  = m_breaks[l-1] 
            IC_vector[l] = IC_vector[l-1]  
    IC_min        = np.min(IC_vector)
    lambd_number = np.argmin(IC_vector)
    lambd_star   = lambd_values[lambd_number]
    m_star       = m_breaks[lambd_number]
    return IC_vector,m_breaks,IC_min,lambd_number,lambd_star,m_star
      

      
#Dimensions
p   = 4
n   = 25
T   = 5
m   = 1
r   = 5


# #Getting tunning parameter
# data = DATA3(r,m,T,n,p,0,0,0)
# X,y,beta,u,eps,F,y_tilde,u_tilde,X_mean,X_tilde= data.DGPO()
# lambd_values = np.logspace(-3, 3, 50)
# IC_vector,m_breaks,IC_min,lambd_number,lambd_star,m_star=  IC(lambd_values,y_tilde,X_tilde,p+1,T-1,n)
# opt = Optimize(p+1,T-1,n)
# b_w = opt.OLS(X_tilde, y_tilde)[0] 

# #getting breaks,beta values
# b_lstar,m_lstar,_ ,_ = opt.FGLS(X_tilde,y_tilde,b_w,lambd_star)
# print("These are the given betas \n {}".format(tabulate(beta)))
# print("These are the estimated betas \n {}".format(tabulate(b_lstar)))
# print("\n")
# print("We had m = {} true breaks and m_est = {} estimated".format(m,m_lstar))

# #PLOTTING
# fig,ax1 = plt.subplots()
# ax1.plot(lambd_values, IC_vector, color="blue", marker=".")
# ax1.set_xlabel("Tuning parmeter",fontsize=12)
# ax1.set_ylabel("IC",color="blue",fontsize=12)
# plt.xscale("log")
# ax2=ax1.twinx() 
# ax2.plot(lambd_values, m_breaks,color="green",marker=".")
# ax2.set_ylabel("Number of breaks",color="green",fontsize=12)
# ax2.spines["left"].set_color("blue")
# ax2.spines["right"].set_color("green")
# ax1.tick_params(axis='y', colors='blue')
# ax1.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.tick_params(axis='y', colors='green')
# plt.savefig('img232.pdf')
# plt.show()

def main():
    # # #MC
    lambd_values = np.logspace(-3, 3, 50)
    opt = Optimize(p+1,T-1,n)
    sim =1000
    b_lsim = np.zeros((p+1,T-1))
    m_sim  = 0
    m_freq = 0
    T_freq = 0
    Failed = 0
    for k in range(sim):
        try:
            data = DATA3(r,m,T,n,p,0,0,0)
            X,y,beta,u,eps,F,y_tilde,u_tilde,X_mean,X_tilde= data.DGP1()
            IC_vector,m_breaks,IC_min,lambd_number,lambd_star,m_star =  IC(lambd_values,y_tilde,X_tilde,p+1,T-1,n)
            b_w = opt.OLS(X_tilde, y_tilde)[0]
            b_lstar,m_lstar,_ ,_ = opt.FGLS(X_tilde,y_tilde,b_w,lambd_star)
            print(m_lstar)
            if m_lstar != m:
                m_freq += 1
            if m_lstar == m:
                if m==0: ## no need to do anything, there is no jump.
                    pass
                if m==1: ## if the jump is less than then there is no jump and add to T_freq.
                    if np.linalg.norm(b_lstar[0:(p+1),math.floor((T)/(m+1))-2]- b_lstar[0:(p+1),math.floor((T)/(m+1))-1]) < 0.01:
                        T_freq += 1
                if m==2:
                    if np.linalg.norm(b_lstar[0:(p+1),math.floor((T-1)/(m+1))-2]- b_lstar[0:(p+1),math.floor((T-1)/(m+1))-1]) < 0.01:
                        T_freq += 1
                    elif np.linalg.norm(b_lstar[0:(p+1),math.floor(2*((T-1)/(m+1)))-2] - b_lstar[0:(p+1),math.floor(2*((T-1)/(m+1)))-1] )<0.01:
                        T_freq += 1
            b_lsim = b_lsim + b_lstar
            m_sim += m_lstar
        except:
            Failed +=1
        
    

    b_lsim = b_lsim/(sim-Failed)
    m_sim = m_sim/(sim-Failed)
    error = 1/(beta.shape[0]*beta.shape[1])*(np.linalg.norm(beta - b_lsim,"fro"))

    print("\n")
    print("These are the given betas \n {}".format(tabulate(beta)))
    print("\n")
    print("These are the estimated betas \n {}".format(tabulate(b_lsim)))
    print("\n")
    print("We had m = {} true breaks, m_est_average = {} estimated , {} frequency of false break estimation and {} frequence of false break date estimation ".format(m,m_sim,m_freq,T_freq))
    print("\n")
    print("the dimensions are  p ={},T={},r={}, n={} and sim ={}".format(p,T,r,n,sim))
    print("\n")
    print("The MFE = {}".format(error))

if __name__ == "__main__":
    main()
