import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import pdb

# basic parameters
dt = 0.1*60*60  # delta t [s]
tn = 1000  # number of time steps

L = 5e6  # length of basin [m]
l = 1.  # basin unit length [bd]
xi = int(1e4+1)  # number of horizontal grids
dx = l/(xi-1.)  # grid length in terms of bd [bd]

beta = 1e-4  # Rossby parameter [bd-1s-1]
gamma = 1e-6  # friction coefficient [s-1]
wb_width = gamma/beta  # west boundary layer width [bd]
wc = -1e-4  # wind curl [s-2]
vbar = wc/beta  # average v-velocity without friction at steady state (beta*v=wc) [bd s-1]

# methodology switchs
bc = 'noslip'  # noslip or freeslip
dmethod = 'mixed'  # forward, leapfrog, mixed, or backward

# --------------------------------------------------------------------
# initiate variables
# horizontal location
# add 2 imaginary points, one at each end to deal with boundary condition
x = np.linspace(-dx, l+dx, xi+2)  # [db]

# stream funciton
phi = np.zeros((tn, xi+2))  # [db2s-1]
# v-velocity
v = np.zeros((tn, xi+2))  # [db s-1]
# vorticity
zeta = np.zeros((tn, xi+2))  # [s-1]

# initial condition (if not zero)
phi[0, :] = np.zeros(xi+2)  # [db2s-1]

# boundary condition
# no penetration
phiw = np.zeros(tn)  # west
phie = np.zeros(tn)  # east
if bc=='noslip':
    # no slip
    vw = np.zeros(tn)  # west
    ve = np.zeros(tn)  # east
elif bc=='freeslip':
    # free slip
    zetaw = np.zeros(tn)  # west
    zetae = np.zeros(tn)  # east

# --------------------------------------------------------------------
# iterate to solve the equation

# construct linear equation matrix
A = -2*sp.eye(xi+2) + \
    sp.eye(xi+2, k=1) + \
    sp.eye(xi+2, k=-1)
# setup boundary condition
# no penetration
A[1, 0], A[1, 2], A[-2, -1], A[-2, -3] = 0, 0, 0, 0
A[1, 1], A[-2, -2] = 1, 1
# second order
if bc=='noslip':
    A[0, 0] = -1
    A[0, 2] = 1
    A[-1, -3] = -1
    A[-1, -1] = 1
    A[0, 1] = 0
    A[-1, -2] = 0
elif bc=='freeslip':
    A[0, 0] = 1
    A[0, 2] = 1
    A[-1, -3] = 1
    A[-1, -1] = 1
    A[0, 1] = -2
    A[-1, -2] = -2

plt.figure()
plt.xlim(0, 1)
plt.show(block=False)

for n in range(tn-1):
# for n in range(1):
    
    # claculate each term in the equation except d/dt
    # sverdrup
    sv = -beta*v[n, 2:-2]
    # friction
    fc = -gamma*zeta[n, 2:-2]

    if n==0:
        # first step always use forward scheme
        B = np.zeros(xi+2)
        B[2:-2] = phi[n, 3:-1] + phi[n, 1:-3] - 2*phi[n, 2:-2] + \
                  2*dt*dx*dx*(sv + wc + fc)
        # setup boundary condition
        # no penetration
        B[1] = phiw[n]
        B[-2] = phie[n]
        # second order
        if bc=='noslip':
            B[0] = 2*dx*vw[n]
            B[-1] = 2*dx*ve[n]
        elif bc=='freeslip':
            B[0] = dx**2*zetaw[n]
            B[-1] = dx**2*zetae[n]
            
        # solve linear equation
        phi[n+1, :] = np.linalg.solve(A, B)

        # update v and zeta
        v[n+1, 1:-1] = (phi[n+1, 2:] - phi[n+1, :-2]) / (2*dx)
        zeta[n+1, 1:-1] = (phi[n+1, 2:] + phi[n+1, :-2] - 2*phi[n+1, 1:-1]) / (dx**2)
        continue
    
    if (dmethod=='forward') | (dmethod=='mixed'):
        B = np.zeros(xi+2)
        B[2:-2] = phi[n, 3:-1] + phi[n, 1:-3] - 2*phi[n, 2:-2] + \
                  2*dt*dx*dx*(sv + wc + fc)
        # setup boundary condition
        # no penetration
        B[1] = phiw[n]
        B[-2] = phie[n]
        # second order
        if bc=='noslip':
            B[0] = 2*dx*vw[n]
            B[-1] = 2*dx*ve[n]
        elif bc=='freeslip':
            B[0] = dx**2*zetaw[n]
            B[-1] = dx**2*zetae[n]

        # solve linear equation
        phi1fw = np.linalg.solve(A, B)

    if (dmethod=='leapfrog') | (dmethod=='mixed'):
        B = np.zeros(xi+2)
        B[2:-2] = phi[n-1, 3:-1] + phi[n-1, 1:-3] - 2*phi[n-1, 2:-2] + \
                  2*dt*dx*dx*(sv + wc - fc)
        # setup boundary condition
        # no penetration
        B[1] = phiw[n]
        B[-2] = phie[n]
        # second order
        if bc=='noslip':
            B[0] = 2*dx*vw[n]
            B[-1] = 2*dx*ve[n]
        elif bc=='freeslip':
            B[0] = dx**2*zetaw[n]
            B[-1] = dx**2*zetae[n]

        phi1lf = np.linalg.solve(A, B)

    if (dmethod=='forward'):
        phi[n+1, :] = phi1fw
    if (dmethod=='leapfrog'):
        phi[n+1, :] = phi1lf
    if (dmethod=='mixed'):
        phi[n+1, :] = 0.5*(phi1fw+phi1lf)

    # update v and zeta
    v[n+1, 1:-1] = (phi[n+1, 2:] - phi[n+1, :-2]) / (2*dx)
    zeta[n+1, 1:-1] = (phi[n+1, 2:] + phi[n+1, :-2] - 2*phi[n+1, 1:-1]) / (dx**2)

    # pdb.set_trace()
    if (n % 50 == 0):
        plt.plot(x, v[n+1, :])
        plt.draw()

# plt.plot(phi)
# plt.show(block=False)

