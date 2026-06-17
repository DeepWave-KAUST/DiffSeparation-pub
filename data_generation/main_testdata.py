"""
Author: Shijun Cheng
Email: sjcheng.academic@gmail.com
Description: This script uses numba for acceleration to generate training data for elastic wave separation.
Reference: Implementation based on Zhu's paper.
Paper URL: Zhu, H. (2017). Elastic wavefield separation based on the Helmholtz decomposition. Geophysics, 82(2), S173-S183.
"""

import numpy as np
import matplotlib.pyplot as plt
from cpml import cpml_coefficients, initialize_cpml
from source import source
from updatep import updatep1_cuda, updatep2_cuda
from updatev import updatev1_cuda, updatev2_cuda
from calmid_cuda import calmid1_cuda, calmid2_cuda
import scipy.io as sio
from numba import cuda
import math
import random
from scipy.ndimage import gaussian_filter
import os

def generate_random_integers(start, end, count, min_difference=150):
    """
    Generate a list of 'count' random integers between 'start' and 'end'
    ensuring that the difference between any two numbers is at least 'min_difference'.
    """
    numbers = []
    while len(numbers) < count:
        num = random.randint(start, end)
        # Check if the new number satisfies the minimum difference condition with all existing numbers
        if all(abs(num - existing) >= min_difference for existing in numbers):
            numbers.append(num)
    return numbers


# --------------------------
# Parameter settings
# --------------------------
npd = 40  # Number of grid points for the CPML (damping) region
tmax = 1.3  # Maximum propagation time (in seconds)
Nx, Nz = 256, 256  # Number of grid points in x and z directions
dx, dz = 10, 10  # Spatial grid spacing in x and z directions
dt = 0.001  # Time sampling interval (in seconds)
favg = 12  # Central frequency of the source wavelet
kk = 5  # Finite difference accuracy order (number of coefficients)

# --------------------------
# Finite difference coefficients
# --------------------------
coef = [1.2112427, -0.08972168, 0.013842773, -0.0017656599, 0.00011867947]
coef_d = cuda.to_device(coef)  # Transfer the coefficients to the GPU

# --------------------------
# Density model setup
# --------------------------
rho0 = np.ones((Nz, Nx))  # Create an initial density model with all values set to 1
rho = np.pad(rho0, npd, mode='edge')  # Pad the density model to account for the CPML boundary
rho_d = cuda.to_device(rho)  # Transfer the padded density model to the GPU

# --------------------------
# Seismic source generation
# --------------------------
Nt = int(tmax / dt) + 1  # Calculate the total number of time steps
pfac = 10  # Source scaling factor
sour = source(pfac, favg, Nt, dt)  # Generate the seismic source time function
sour_d = cuda.to_device(sour)  # Transfer the source to the GPU

# --------------------------
# Initialize wavefield and memory variables
# --------------------------
shape = (Nz + 2 * npd, Nx + 2 * npd)  # Total grid shape including CPML boundaries

# Velocity fields (vertical and horizontal components)
vz = np.zeros(shape, dtype=np.float32)
vx = np.zeros(shape, dtype=np.float32)

# Stress fields
pxx = np.zeros(shape, dtype=np.float32)
pzz = np.zeros(shape, dtype=np.float32)
pxz = np.zeros(shape, dtype=np.float32)

# Memory variables for the absorbing boundary (CPML)
s_px = np.zeros(shape, dtype=np.float32)
s_pz = np.zeros(shape, dtype=np.float32)
s_sx = np.zeros(shape, dtype=np.float32)
s_sz = np.zeros(shape, dtype=np.float32)
tau_p = np.zeros(shape, dtype=np.float32)
tau_s = np.zeros(shape, dtype=np.float32)

# Intermediate variables for wavefield separation
mid_vxp = np.zeros(shape, dtype=np.float32)
mid_vzp = np.zeros(shape, dtype=np.float32)
mid_vxs = np.zeros(shape, dtype=np.float32)
mid_vzs = np.zeros(shape, dtype=np.float32)

# CPML-related variables for stress and velocity fields
pml_pxxx = np.zeros(shape, dtype=np.float32)
pml_pxzz = np.zeros(shape, dtype=np.float32)
pml_pxzx = np.zeros(shape, dtype=np.float32)
pml_pzzz = np.zeros(shape, dtype=np.float32)
pml_vxx = np.zeros(shape, dtype=np.float32)
pml_vzz = np.zeros(shape, dtype=np.float32)
pml_vxz = np.zeros(shape, dtype=np.float32)
pml_vzx = np.zeros(shape, dtype=np.float32)
pml_tau_px = np.zeros(shape, dtype=np.float32)
pml_tau_pz = np.zeros(shape, dtype=np.float32)
pml_tau_sx = np.zeros(shape, dtype=np.float32)
pml_tau_sz = np.zeros(shape, dtype=np.float32)

# Additional intermediate variables for stress and memory
xxx = np.zeros(shape, dtype=np.float32)
xzz = np.zeros(shape, dtype=np.float32)
xzx = np.zeros(shape, dtype=np.float32)
zzz = np.zeros(shape, dtype=np.float32)
tau_px = np.zeros(shape, dtype=np.float32)
tau_pz = np.zeros(shape, dtype=np.float32)
tau_sx = np.zeros(shape, dtype=np.float32)
tau_sz = np.zeros(shape, dtype=np.float32)

# --------------------------
# CUDA kernel grid setup for time stepping
# --------------------------
threads_per_block = (16, 16)  # Define the number of threads per block for CUDA kernels
blocks_per_grid_x = int(math.ceil(shape[0] / threads_per_block[0]))  # Calculate the number of blocks in x direction
blocks_per_grid_y = int(math.ceil(shape[1] / threads_per_block[1]))  # Calculate the number of blocks in y direction
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)  # Combine into grid dimensions

# --------------------------
# Simulation shot parameters
# --------------------------
nshot = 2  # Number of shots per velocity model
nsnaps = 5  # Number of snapshots to save per shot

# --------------------------
# Load training velocity models
# --------------------------
vp_test = np.load('../dataset/velocity_model/vp_test.npy')  # Load P-wave velocity models
vs_test = np.load('../dataset/velocity_model/vs_test.npy')  # Load S-wave velocity models
data_num = vp_test.shape[0]  # Total number of velocity models in the dataset

# --------------------------
# Create training dataset file folder
# --------------------------
dir_test = f'../dataset/test/'
os.makedirs(dir_test, exist_ok=True)

# Loop over each velocity model in the dataset
for data_id in range(data_num):
    vp0 = vp_test[data_id]
    vs0 = vs_test[data_id]
    # Apply Gaussian smoothing to the velocity models
    vp0 = gaussian_filter(vp0, 2)
    vs0 = gaussian_filter(vs0, 2)
    # Pad the velocity models to include CPML boundaries
    vp = np.pad(vp0, npd, mode='edge')
    vs = np.pad(vs0, npd, mode='edge')

    # --------------------------
    # Compute elastic parameters
    # --------------------------
    # Lamb's parameter (lambda) and shear modulus (mu) are computed from velocities and density
    lamb = rho * (vp**2 - 2 * vs**2)
    muon = rho * vs**2

    lamb_d = cuda.to_device(lamb)  # Transfer lambda to GPU
    muon_d = cuda.to_device(muon)  # Transfer mu to GPU

    # --------------------------
    # Initialize CPML boundary parameters
    # --------------------------
    vp_max = vp0.max()  # Maximum P-wave velocity in the current model
    # Initialize CPML parameters (damping coefficients and scaling factors)
    ax, az, bx, bz, kxc, kzc = initialize_cpml(Nx, Nz, npd, vp_max, dx, favg, dt)

    # Transfer CPML parameters to the GPU
    ax_d = cuda.to_device(ax)
    az_d = cuda.to_device(az)
    bx_d = cuda.to_device(bx)
    bz_d = cuda.to_device(bz)
    kxc_d = cuda.to_device(kxc)
    kzc_d = cuda.to_device(kzc)

    # --------------------------
    # Loop over simulation shots
    # --------------------------
    for ishot in range(nshot):
       print(f'forward modeling for data {data_id} shot {ishot + 1}')
       # Set source location:
       # For the first shot, set depth to 0 and choose a random x position;
       # for subsequent shots, choose random positions within specified bounds.
       if ishot == 0:
          nsz = 0
          nsx = 128
       else:
          nsz = 128
          nsx = 128
       
       # --------------------------
       # Copy initial fields and memory variables to the GPU for this shot
       # --------------------------
       vz_d = cuda.to_device(vz)
       vx_d = cuda.to_device(vx)
       pxx_d = cuda.to_device(pxx)
       pzz_d = cuda.to_device(pzz)
       pxz_d = cuda.to_device(pxz)
       s_px_d = cuda.to_device(s_px)
       s_pz_d = cuda.to_device(s_pz)
       s_sx_d = cuda.to_device(s_sx)
       s_sz_d = cuda.to_device(s_sz)
       tau_p_d = cuda.to_device(tau_p)
       tau_s_d = cuda.to_device(tau_s)

       mid_vxp_d = cuda.to_device(mid_vxp)
       mid_vzp_d = cuda.to_device(mid_vzp)
       mid_vxs_d = cuda.to_device(mid_vxs)
       mid_vzs_d = cuda.to_device(mid_vzs)

       pml_pxxx_d = cuda.to_device(pml_pxxx)
       pml_pxzz_d = cuda.to_device(pml_pxzz)
       pml_pxzx_d = cuda.to_device(pml_pxzx)
       pml_pzzz_d = cuda.to_device(pml_pzzz)
       pml_vxx_d = cuda.to_device(pml_vxx)
       pml_vzz_d = cuda.to_device(pml_vzz)
       pml_vxz_d = cuda.to_device(pml_vxz)
       pml_vzx_d = cuda.to_device(pml_vzx)
       pml_tau_px_d = cuda.to_device(pml_tau_px)
       pml_tau_pz_d = cuda.to_device(pml_tau_pz)
       pml_tau_sx_d = cuda.to_device(pml_tau_sx)
       pml_tau_sz_d = cuda.to_device(pml_tau_sz)

       xxx_d = cuda.to_device(xxx)
       xzz_d = cuda.to_device(xzz)
       xzx_d = cuda.to_device(xzx)
       zzz_d = cuda.to_device(zzz)
       tau_px_d = cuda.to_device(tau_px)
       tau_pz_d = cuda.to_device(tau_pz)
       tau_sx_d = cuda.to_device(tau_sx)
       tau_sz_d = cuda.to_device(tau_sz)

       # --------------------------
       # Generate random time indices to save snapshots
       # --------------------------
       random_it = [200, 400, 600, 800, 1000]
       snap_id = 1  # Snapshot counter

       # --------------------------
       # Time stepping loop for simulation
       # --------------------------
       for it in range(Nt):
           # Add the seismic source to the vertical velocity field at the source location
           vz_d[nsz + npd, nsx + npd] += sour_d[it]

           # Update velocity fields (first CUDA kernel)
           updatev1_cuda[blocks_per_grid, threads_per_block](
               shape, dx, dz, vz_d, vx_d, pxx_d, pzz_d, pxz_d, rho_d, dt, coef_d, kk,
               ax_d, az_d, bx_d, bz_d, kxc_d, kzc_d, pml_pxxx_d, pml_pxzz_d, pml_pxzx_d,
               pml_pzzz_d, tau_p_d, s_px_d, s_pz_d, pml_tau_px_d, pml_tau_pz_d,
               tau_s_d, s_sx_d, s_sz_d, pml_tau_sx_d, pml_tau_sz_d,
               xxx_d, xzz_d, xzx_d, zzz_d, tau_px_d, tau_pz_d, tau_sx_d, tau_sz_d
           )

           # Update velocity fields (second CUDA kernel)
           updatev2_cuda[blocks_per_grid, threads_per_block](
               shape, dx, dz, vz_d, vx_d, pxx_d, pzz_d, pxz_d, rho_d, dt, coef_d, kk,
               ax_d, az_d, bx_d, bz_d, kxc_d, kzc_d, pml_pxxx_d, pml_pxzz_d, pml_pxzx_d,
               pml_pzzz_d, tau_p_d, s_px_d, s_pz_d, pml_tau_px_d, pml_tau_pz_d,
               tau_s_d, s_sx_d, s_sz_d, pml_tau_sx_d, pml_tau_sz_d,
               xxx_d, xzz_d, xzx_d, zzz_d, tau_px_d, tau_pz_d, tau_sx_d, tau_sz_d
           )

           # Compute intermediate variables for wavefield separation using CUDA kernels
           calmid1_cuda[blocks_per_grid, threads_per_block](shape, dx, dz, coef_d, kk, vx_d, vz_d, 
                   xxx_d, xzz_d, xzx_d, zzz_d, tau_px_d, tau_pz_d
           )

           calmid2_cuda[blocks_per_grid, threads_per_block](shape, dx, dz, coef_d, kk, vx_d, vz_d, 
                   tau_px_d, tau_pz_d, 
                   mid_vxp_d, mid_vzp_d, mid_vxs_d, mid_vzs_d
           )

           # Update stress fields (first CUDA kernel)
           updatep1_cuda[blocks_per_grid, threads_per_block](
               shape, dx, dz, pxx_d, pzz_d, pxz_d, vz_d, vx_d, lamb_d, muon_d, dt,
               coef_d, kk, ax_d, az_d, bx_d, bz_d, kxc_d, kzc_d, pml_vxx_d, pml_vzz_d,
               pml_vxz_d, pml_vzx_d, tau_p_d, tau_s_d,
               xxx_d, xzz_d, xzx_d, zzz_d
           )

           # Update stress fields (second CUDA kernel)
           updatep2_cuda[blocks_per_grid, threads_per_block](
               shape, dx, dz, pxx_d, pzz_d, pxz_d, vz_d, vx_d, lamb_d, muon_d, dt,
               coef_d, kk, ax_d, az_d, bx_d, bz_d, kxc_d, kzc_d, pml_vxx_d, pml_vzz_d,
               pml_vxz_d, pml_vzx_d, tau_p_d, tau_s_d,
               xxx_d, xzz_d, xzx_d, zzz_d
           )

           # Save snapshots at selected time steps
           if it in random_it:
               # Copy wavefields from GPU to host memory (excluding CPML boundaries)
               snapvx = vx_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
               snapvz = vz_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
               snapvx_p = s_px_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
               snapvz_p = s_pz_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
               snapvx_s = s_sx_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
               snapvz_s = s_sz_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
               snap_mid_vxp = mid_vxp_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
               snap_mid_vzp = mid_vzp_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
               snap_mid_vxs = mid_vxs_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
               snap_mid_vzs = mid_vzs_d[npd:npd + Nz, npd:npd + Nx].copy_to_host()
      
               # Save the snapshot data along with model parameters and source position into a .mat file
               sio.savemat(f'{dir_test}data{data_id}_shot{ishot+1}_snap{snap_id}.mat', {
                   'vp': vp0, 'vs': vs0, 'snapit': it, 'sz': nsz, 'sx': nsx, 
                   'vx': snapvx, 'vz': snapvz, 'vx_p': snapvx_p, 'vz_p': snapvz_p, 
                   'vx_s': snapvx_s, 'vz_s': snapvz_s,
                   'mid_vxp': snap_mid_vxp, 'mid_vzp': snap_mid_vzp, 
                   'mid_vxs': snap_mid_vxs, 'mid_vzs': snap_mid_vzs
               })
               snap_id += 1

print("Simulation complete.")
