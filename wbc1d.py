import numpy as np

# basic parameters
dt = 24*60*60  # delta t [s]
tn = 1000  # number of time steps

L = 5e6  # length of basin [m]
l = 1  # basin unit length [bd]
xi = 1e4+1  # number of horizontal grids
dx = l/(xi-1)  # grid length in terms of bd [bd]

beta = 1e-4  # Rossby parameter [bd-1s-1]
gamma = 1e-6  # friction coefficient [s-1]
wb_width = gamma/beta  # west boundary layer width [bd]
wc = -1e-4  # wind curl [s-2]
vbar = wc/beta  # average v-velocity without friction at steady state (beta*v=wc) [bd s-1]

# methodology switchs
bc = 'noslip'  # noslip or freeslip
dis = 'forward'  # forward, leapfrog, mixed, or backward

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
# # no penetration
# phiw = np.zeros(tn)  # west
# phie = np.zeros(tn)  # east
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

for n in range(tn-1):

    phi_n_x = (phi[n, 2:] - phi[n, :-2]) / (2*dx)
    phi_n_xx = (phi[n, 2:] + phi[n, :-2] - 2*phi[n, 1:-1]) / (dx**2)

    if n==0:
        # first step always use forward scheme
        B = np.zeros(xi+2)
        B[1:-1] =   phi[n, 2:] + phi[n, :-2] - 2*phi[n, 1:-1]
                  + 2*dt*dx*dx*(-beta*phi_n_x + wc - gamma*phi_n_xx)
        B[0] = 2*dx*vw[n]
        B[-1] = 2*dx*ve[n]
