import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# basic parameters
dt = 0.2*60*60  # delta t [s]
tn = 10000  # number of time steps
t = np.arange(0, dt*tn, dt)

D = 200.  # water column depth [m]
zi = int(2e3+1)  # number of vertical grids
dz = D/(zi-1.)  # grid length [m]

f = 1.0e-4  # Coriolis parameter [s-1]
Az = 1.0e-2  # viscosity [m2s-1]
tao = 1.0e-1  # wind stress [pa]

# methodology switchs
bc = 'noslip'  # noslip or freeslip
dmethod = 'trape'  # forward, leapfrog, trape, or AB

# --------------------------------------------------------------------
# initiate variables
# vertical grid
# add 2 imaginary points, one at each end to deal with boundary condition
z = np.linspace(-(D+dz), dz, zi+2)  # [m]

# velocity
v = np.zeros((tn, zi+2))  # [db s-1]

