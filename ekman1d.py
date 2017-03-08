import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# basic parameters
dt = 0.2*60*60  # delta t [s]
tn = 5000  # number of time steps
t = np.arange(0, dt*tn, dt)

D = 200.  # water column depth [m]
zi = int(2e3)  # number of vertical grids
dz = -D/(zi*1.)  # grid length [m]

f = 1.0e-4  # Coriolis parameter [s-1]
Az0 = 1.0e-2  # viscosity [m2s-1]
rho0 = 1.0e3  # water density [kg m-3]
tauw = 1.0e-2/rho0  # surface wind stress [m2s-2]
tauf = 1.0e-3/rho0  # bottom friction stress [m2s-2]
pgf0 = -3.0e-3/rho0  # pressure gradient force [m s-2]

DE = np.pi*np.sqrt(2*Az0/f)  # average Ekman depth [m]
vg = pgf0/f  # geostrophic velocity [m s-1]

# methodology switchs
dmethod = 'trape'  # forward, backward, leapfrog, trape, or AB

# --------------------------------------------------------------------
# initiate variables
# vertical grid
# add 2 imaginary points, one at each end to deal with boundary condition
z = np.linspace(-0.5*dz, -(D-0.5*dz), zi+2)  # [m]
z_tau = 0.5*(z[1:] + z[:-1])

# at tau points
# viscosity
Az = np.ones(zi+1)*Az0  # [m2s-1]

# at rho points
# velocity
v = np.zeros((tn, zi+2), dtype=np.complex_)  # [m s-1]
# friction
fric = np.zeros((tn, zi+2), dtype=np.complex_)  # [m s-2]
# stress
tau = np.zeros((tn, zi+2), dtype=np.complex_)  # [m2s-2]

# initial condition (if not zero)
v[0, :] = 0  # [m s-1]
fric[0, 1:-1] = (Az[1:]*v[0, 2:] + \
                 -(Az[1:]+Az[:-1])*v[0, 1:-1] + \
                 Az[:-1]*v[0, :-2] \
                )/(dz*dz)

# boundary condition
# surface
taus = np.ones(tn)*tauw  # [m2s-2]
# bottom
vb = np.zeros(tn)  # [m s-1]
# taub = np.ones(tn)*tauf  # [m2s-2]

# forcing
# surface
pgf = np.ones((tn, zi+2))*pgf0  # []m s-2

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

# first step always trapezoidal
alpd0 = 1.
betd0 = 0.
gamd0 = 0.5
deld0 = 0.5
epsd0 = 0.

# --------------------------------------------------------------------
# construct linear equation matrix
diag0 = np.zeros(zi+2, dtype=np.complex_)
diagp1 = np.zeros(zi+1)
diagm1 = np.zeros(zi+1)
diag0[1:-1] = 1. + dt*gamd*(f*1j + (Az[1:]+Az[:-1])/(dz*dz))
diagp1[1:] = -dt*gamd*Az[1:]/(dz*dz)
diagm1[:-1] = -dt*gamd*Az[:-1]/(dz*dz)

A = sparse.diags([diag0, diagp1, diagm1], [0, 1, -1]).tocsr()

# setup boundary condtion
# surface match wind stress
A[0, 0] = -1
A[0, 1] = 1
# bottom non-slip
A[-1, -2], A[-1, -1] = 0.5, 0.5

# for the first step
# construct linear equation matrix
diag0 = np.zeros(zi+2, dtype=np.complex_)
diagp1 = np.zeros(zi+1)
diagm1 = np.zeros(zi+1)
diag0[1:-1] = 1. + dt*gamd0*(f*1j + (Az[1:]+Az[:-1])/(dz*dz))
diagp1[1:] = -dt*gamd0*Az[1:]/(dz*dz)
diagm1[:-1] = -dt*gamd0*Az[:-1]/(dz*dz)

A0 = sparse.diags([diag0, diagp1, diagm1], [0, 1, -1]).tocsr()

# setup boundary condtion
# surface match wind stress
A0[0, 0] = -1
A0[0, 1] = 1
# bottom noslip
A0[-1, -2], A0[-1, -1] = 0.5, 0.5

# iterate to solve the equation
for n in range(tn-1):
   
    if n==0:
        # first step always use trapezoidal scheme
        B = np.zeros(zi+2, dtype=np.complex_)
        B[1:-1] = alpd0*v[n, 1:-1] + \
                  dt*(gamd0*pgf[n, 1:-1] + \
                      deld0*(-f*1j*v[n, 1:-1] + fric[n, 1:-1] - pgf[n, 1:-1]) \
                     )

        # setup boundary condition
        # surface match wind stress
        B[0] = taus[n]/Az[0]
        # bottom non-slip
        B[-1] = vb[n]

        # solve linear equation
        v[n+1, :] = spsolve(A0, B)

        # update fric
        fric[n+1, 1:-1] = (Az[1:]*v[n+1, 2:] + \
                           -(Az[1:]+Az[:-1])*v[n+1, 1:-1] + \
                           Az[:-1]*v[n+1, :-2] \
                          )/(dz*dz)
        continue
    
    # --------------------------------------------------------------------

    B = np.zeros(zi+2, dtype=np.complex_)
    B[1:-1] = alpd*v[n, 1:-1] + \
            betd*v[n-1, 1:-1] + \
              dt*(gamd*(-pgf[n+1, 1:-1]) + \
                  deld*(-f*1j*v[n, 1:-1] + fric[n, 1:-1] - pgf[n+1, 1:-1]) + \
                  epsd*(-f*1j*v[n-1, 1:-1] + fric[n-1, 1:-1] - pgf[n-1, 1:-1]) \
                 )

    # setup boundary condition
    # surface match wind stress
    B[0] = taus[n]/Az[0]
    # bottom non-slip
    B[-1] = vb[n]

    # solve linear equation
    v[n+1, :] = spsolve(A, B)

    # update fric
    fric[n+1, 1:-1] = (Az[1:]*v[n+1, 2:] + \
                       -(Az[1:]+Az[:-1])*v[n+1, 1:-1] + \
                       Az[:-1]*v[n+1, :-2] \
                      )/(dz*dz)

# make plots
# interactive
pltv = 1
if pltv == 1:
    # plot v interactively
    plt.figure()
    plt.show(block=False)
    for i in range(len(t)):
        if (i % 10 == 0):
            plt.gca().cla()
            plt.plot(v[i, :].real, z)
            plt.plot(v[i, :].imag, z)
            plt.xlim(-0.06, 0.06)
            plt.draw()

# Ekman spiral
plt_spiral = 0
if plt_spiral == 1:
    plt.figure()
    plt.plot(v[-1, :].imag, v[-1, :].real)
    plt.savefig('ekman_spiral.png', format='png', dpi=900)
    plt.close()

# hovmoller
plt_hov = 0
if plt_hov == 1:
    plt.figure()
    plt.pcolor(t[::5]/24/60/60, z[::10], v[::5, ::10].real.T, cmap=plt.cm.RdYlBu_r)
    plt.xlabel('Days')
    plt.ylabel('Depth')
    plt.xlim(0, t[-1]/24/60/60)
    plt.ylim(-D, 0)
    plt.clim(-0.06, 0.06)
    cb = plt.colorbar()
    cb.ax.set_ylabel(r'U Velocity')
    plt.savefig('u_' + dmethod + '.png', format='png', dpi=900)
    plt.close()

    plt.figure()
    plt.pcolor(t[::5]/24/60/60, z[::10], v[::5, ::10].imag.T, cmap=plt.cm.RdYlBu_r)
    plt.xlabel('Days')
    plt.ylabel('Depth')
    plt.xlim(0, t[-1]/24/60/60)
    plt.ylim(-D, 0)
    plt.clim(-0.06, 0.06)
    cb = plt.colorbar()
    cb.ax.set_ylabel(r'U Velocity')
    plt.savefig('v_' + dmethod + '.png', format='png', dpi=900)
    plt.close()

