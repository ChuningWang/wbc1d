import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# basic parameters
dt = 0.2*60*60  # delta t [s]
tn = 10000  # number of time steps
t = np.arange(0, dt*tn, dt)

L = 5.0e6  # length of basin [m]
l = 1.  # basin unit length [bd]
xi = int(5e3+1)  # number of horizontal grids
dx = l/(xi-1.)  # grid length in terms of bd [bd]

beta = 1.0e-4  # Rossby parameter [bd-1s-1]
gamma = 1.0e-6  # friction coefficient [s-1]
wb_width = gamma/beta  # west boundary layer width [bd]
wc = -1e-4  # wind curl [s-2]
vbar = wc/beta  # average v-velocity without friction at steady state (beta*v=wc) [bd s-1]

# methodology switchs
bc = 'noslip'  # noslip or freeslip
dmethod = 'leapfrog'  # forward, leapfrog, trape, or AB

# --------------------------------------------------------------------
# initiate variables
# horizontal location
# add 2 imaginary points, one at each end to deal with boundary condition
x = np.linspace(-dx/l, 1+dx/l, xi+2)  # [db]

# stream funciton
phi = np.zeros((tn, xi+2))  # [db2s-1]
# v-velocity
v = np.zeros((tn, xi+2))  # [db s-1]
# vorticity
zeta = np.zeros((tn, xi+2))  # [s-1]

# initial condition (if not zero)
phi[0, :] = np.zeros(xi+2)  # [db2s-1]
v[0, 1:-1] = (phi[0, 2:] - phi[0, :-2]) / (2*dx)
zeta[0, 1:-1] = (phi[0, 2:] + phi[0, :-2] - 2*phi[0, 1:-1]) / (dx**2)

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
A = -2*sparse.eye(xi+2)*(1+dt*gamd*gamma) + \
    sparse.eye(xi+2, k=1)*(1+dt*gamd*(gamma+beta*dx/2)) + \
    sparse.eye(xi+2, k=-1)*(1+dt*gamd*(gamma-beta*dx/2))

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

# first step always use trapezoidal
A0 = -2*sparse.eye(xi+2)*(1+dt*0.5*gamma) + \
     sparse.eye(xi+2, k=1)*(1+dt*0.5*(gamma+beta*dx/2)) + \
     sparse.eye(xi+2, k=-1)*(1+dt*0.5*(gamma-beta*dx/2))
# setup boundary condition
# no penetration
A0[1, 0], A0[1, 2], A0[-2, -1], A0[-2, -3] = 0, 0, 0, 0
A0[1, 1], A0[-2, -2] = 1, 1

# second order
if bc=='noslip':
    A0[0, 0] = -1
    A0[0, 2] = 1
    A0[-1, -3] = -1
    A0[-1, -1] = 1
    A0[0, 1] = 0
    A0[-1, -2] = 0
elif bc=='freeslip':
    A0[0, 0] = 1
    A0[0, 2] = 1
    A0[-1, -3] = 1
    A0[-1, -1] = 1
    A0[0, 1] = -2
    A0[-1, -2] = -2

for n in range(tn-1):
   
    if n==0:
        # first step always use trapezoidal scheme
        B = np.zeros(xi+2)
        B[2:-2] = zeta[n, 2:-2]*dx*dx + \
                  dt*dx*dx*(0.5*wc + \
                            0.5*(-beta*v[n, 2:-2] + wc - gamma*zeta[n, 2:-2]))
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
        phi[n+1, :] = spsolve(A0, B)

        # update v and zeta
        v[n+1, 1:-1] = (phi[n+1, 2:] - phi[n+1, :-2]) / (2*dx)
        zeta[n+1, 1:-1] = (phi[n+1, 2:] + phi[n+1, :-2] - 2*phi[n+1, 1:-1]) / (dx**2)
        continue
    
    # --------------------------------------------------------------------

    B = np.zeros(xi+2)
    B[2:-2] = alpd*zeta[n, 2:-2]*dx*dx + \
              betd*zeta[n-1, 2:-2]*dx*dx + \
              dt*dx*dx*(gamd*wc + \
                        deld*(-beta*v[n, 2:-2] + wc - gamma*zeta[n, 2:-2]) + \
                        epsd*(-beta*v[n-1, 2:-2] + wc - gamma*zeta[n-1, 2:-2]) \
                       )

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
    phi[n+1, :] = spsolve(A, B)
    # update v and zeta
    v[n+1, 1:-1] = (phi[n+1, 2:] - phi[n+1, :-2]) / (2*dx)
    zeta[n+1, 1:-1] = (phi[n+1, 2:] + phi[n+1, :-2] - 2*phi[n+1, 1:-1]) / (dx**2)

# --------------------------------------------------------------------
# make plots

pltv = 0
if pltv == 1:
    # plot v interactively
    plt.figure()
    plt.show(block=False)
    for i in range(len(t)):
        if (i % 100 == 0):
            plt.plot(v[i, :])
            plt.draw()

# plot the output as hovmuller diagram
plt_hov = 1
if plt_hov == 1:
    plt.figure()
    plt.pcolor(t[::5]/24/60/60, x[::10], v[::5, ::10].T, cmap=plt.cm.RdYlBu_r)
    plt.xlabel('Days')
    plt.ylabel('Distance')
    plt.xlim(0, t[-1]/24/60/60)
    plt.ylim(0, 1)
    plt.clim(-5, 5)
    cb = plt.colorbar()
    cb.ax.set_ylabel(r'V Velocity')
    plt.savefig('v_' + bc + '_' + dmethod + '.png', format='png', dpi=900)
    plt.close()




