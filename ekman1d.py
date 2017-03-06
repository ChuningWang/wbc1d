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
zi = int(2e3)  # number of vertical grids
dz = D/(zi*1.)  # grid length [m]

f = 1.0e-4  # Coriolis parameter [s-1]
Az0 = 1.0e-2  # viscosity [m2s-1]
rho0 = 1.0e3  # water density [kg m-3]
taow = 1.0e-2/rho0  # surface wind stress [m2s-2]
taof = 1.0e-3/rho0  # bottom friction stress [m2s-2]
pgf0 = -1.0e-3/rho0  # pressure gradient force [m s-2]

# methodology switchs
dmethod = 'trape'  # forward, leapfrog, trape, or AB

# --------------------------------------------------------------------
# initiate variables
# vertical grid
# add 2 imaginary points, one at each end to deal with boundary condition
z = np.linspace(-(D+0.5*dz), 0.5*dz, zi+2)  # [m]

# at tao points
# viscosity
Az = np.ones(zi+1)*Az0  # [m2s-1]

# at rho points
# velocity
v = np.zeros((tn, zi+2))  # [m s-1]
# stress
tao = np.zeros((tn, zi+2))  # [m2s-2]

# initial condition (if not zero)
v[0, :] = 0  # [m s-1]

# boundary condition
# surface
taos = np.ones(tn)*taow  # [m2s-2]
# bottom
taob = np.ones(tn)*taof  # [m2s-2]

# forcing
# surface
pgf = np.ones(tn)*pgf0  # []m s-2

# choose finite differencing parameters
if dmethod == 'forward':
    alpd = 1.
    betd = 0.
    gamd = 0.
    deld = 1.
    epsd = 0.
elif dmethod == 'backward':
    alpd = 1.
    betd = 0.
    gamd = 1.
    deld = 0.
    epsd = 0.
elif dmethod== 'leapfrog':
    alpd = 0.
    betd = 1.
    gamd = 0.
    deld = 2.
    epsd = 0.
elif dmethod == 'trape':
    alpd = 1.
    betd = 0.
    gamd = 0.5
    deld = 0.5
    epsd = 0.
elif dmethod == 'AB':
    alpd = 1.
    betd = 0.
    gamd = 0.
    deld = 1.5
    epsd = -0.5

# --------------------------------------------------------------------
# iterate to solve the equation

# construct linear equation matrix
diag1 = np.zeros(xi+2)
diag1[1:-1] = 1 + dt*gamd*(f*1j + (Az[1:]+Az[:-1])/(dz*dz))
# A = sparse.eye(xi+2)*(1+dt*gamd*(f*1j+)) + \
#     sparse.eye(xi+2, k=1)*(1+dt*gamd*(gamma+beta*dx/2)) + \
#     sparse.eye(xi+2, k=-1)*(1+dt*gamd*(gamma-beta*dx/2))

