import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import time

import multiprocessing
import itertools

#import autograd.numpy as npa
#from autograd import grad, value_and_grad

import legume
from legume import PlaneWaveExp, GuidedModeExp, Circle, ShapesLayer, Lattice, PhotCryst
from legume.minimize import Minimize

def wg_sc(dx, dy, dr, a, ra, eps_b, d, W, lattice, eps_c):
    # plane-wave expansion parameters
    gmax = 2     # truncation of the plane-wave basis
    Ny = 14      # Number of rows in the y-direction
    Ny_opt = 3   # Number of rows in which the pillars will be modified
    Nx = 1       # Supercell size in the x-direction

    """Define the photonic crystal waveguide given shift parameters
    dx, dy, and dr, for the 2*Nx*Ny_opt number of pillars that get shifted
    """
    phc = PhotCryst(lattice, eps_l = 1, eps_u = 1)
    
    # Initialize a layer and the positions of the pillars for the regular waveguide
    phc.add_layer(d=d, eps_b=eps_b)
    
    xc = []; yc = []
    for ih in range(Ny):
        if ih != Ny//2:
            for ix in range(-Nx//2+1, Nx//2+1):
                xc.append((ih%2)*0.5 + ix)
                if ih <= Ny//2:
                    yc.append((-Ny//2 + ih + (1-W)/2)*np.sqrt(3)/2)
                else:
                    yc.append((-Ny//2 + ih - (1-W)/2)*np.sqrt(3)/2)

    # Add all the pillars, taking care of the shifts
    for ih in range(1, Ny//2+1):
        nx1 = (Ny//2+ih-1)
        nx2 = (Ny//2-ih)
        if ih <= Ny_opt:
            # The ih row includes "optimization" pillars
            for ix in range(Nx):
                circ = Circle(x_cent=xc[nx1*Nx + ix] + dx[(ih-1)*Nx + ix],
                              y_cent=yc[nx1*Nx + ix] + dy[(ih-1)*Nx  + ix],
                              r = ra + dr[(ih-1)*Nx + ix], eps=eps_c)
                phc.add_shape(circ)
                circ = Circle(x_cent=xc[nx2*Nx + ix] + dx[(ih-1+Ny_opt)*Nx  + ix],
                              y_cent=yc[nx2*Nx + ix] + dy[(ih-1+Ny_opt)*Nx  + ix],
                              r = ra + dr[(ih-1+Ny_opt)*Nx + ix], eps=eps_c)
                phc.add_shape(circ)
        else:
            # The ih row includes just regular pillars
            for ix in range(Nx):
                circ = Circle(x_cent = xc[nx2*Nx + ix], y_cent=yc[nx2*Nx + ix], r=ra, eps=eps_c)
                phc.add_shape(circ)
                if ih < Ny//2:
                    circ = Circle(x_cent = xc[nx1*Nx + ix], y_cent=yc[nx1*Nx + ix], r=ra, eps=eps_c)
                    phc.add_shape(circ)

    # Construct and return a plane-wave expansion object
    return phc

def effective_ind(a, diam_nm, n_slab, d_nm, W):
    # plane-wave expansion parameters
    gmax = 2     # truncation of the plane-wave basis
    Ny = 14      # Number of rows in the y-direction
    Ny_opt = 3   # Number of rows in which the pillars will be modified
    Nx = 1       # Supercell size in the x-direction

    # Converting Parameters into Legume units
    ra = 0.5*diam_nm/a       # hole radius
    eps_b = n_slab**2        # slab permittivity (n ~= 3.453 at low temp, n = 3.48 at room temp)
    eps_c = 1                # hole permittivity
    d = d_nm/a               # slab thickness
    
    # Initialize a rectangular lattice
    lattice = Lattice([Nx, 0], [0, (Ny-1+W)*np.sqrt(3)/2])

    # Initialize zero shifts
    dx0 = np.zeros((Nx*2*Ny_opt, ))
    dy0 = np.zeros((Nx*2*Ny_opt, ))
    dr0 = np.zeros((Nx*2*Ny_opt, ))

    # Initialize the PWE and visualize the structure both through the `eps` and the `eps_ft` methods
    phc0 = wg_sc(dx0, dy0, dr0, a, ra, eps_b, d, W, lattice, eps_c)

    nk = 50

    # Define a BZ path in kx
    path = phc0.lattice.bz_path([[0, 0], np.array([np.pi/Nx, 0])], [nk])
    #pwe0.run(kpoints=path['kpoints'], pol='tm', numeig = 150)

    neig = 30

    # Initialize GME
    gme = legume.GuidedModeExp(phc0, gmax=gmax)

    # Set some of the running options
    options = {'gmode_inds': [0], 
               'numeig': neig,
               'verbose': False
                }

    # Run the simulation
    gme.run(kpoints=path['kpoints'], **options)

    f_ind = np.linspace(0, 1, neig)
    k = np.linspace(0, 0.5, nk+1)
    fv, kv = np.meshgrid(f_ind, k)
    # print(kv)
    n_eff = kv/gme.freqs
    wvln = a/gme.freqs

    ind = 14 # Index of mode in question

    n_target = 1.44
    n_diff = n_target - n_eff[:,ind]

    zero_crossings = np.where(np.diff(np.sign(n_diff)))
    cross_wvln = float(wvln[zero_crossings, ind])
    cross_wvlns = np.append(cross_wvlns, cross_wvln)
    
    wvln_target = 1518.37
    wvln_ind = wvln[:,ind]    #wvln for the mode in question
    n_eff_ind = abs(n_eff[:,ind])  #n_eff for the mode in question
    
    sort = np.argsort(wvln_ind) # Sorting wvln monotonically
    
    wvln_sort = wvln_ind[sort]
    n_eff_sort = n_eff_ind[sort]
        
    tck = sc.interpolate.splrep(wvln_sort,n_eff_sort, s=0)
    n_eff_wvln = sc.interpolate.splev(wvln_target, tck, der=0) # Interpolate to find nee at wvln_target
    return n_eff_wvln