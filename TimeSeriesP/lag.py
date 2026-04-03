import numpy as np
from numpy.linalg import matrix_power as power

def Lag(n):
    I_n = np.identity(n-1)
    r_n = np.zeros((1,n-1))
    c_n = np.zeros((n,1)) 
    L_n = np.concatenate((r_n,I_n), axis = 0)
    L = np.concatenate((L_n,c_n), axis = 1)
    return L
