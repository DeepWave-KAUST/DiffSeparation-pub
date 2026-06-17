import numpy as np
from numba import cuda

@cuda.jit
def updatep1_cuda(shape, dx, dz, pxx, pzz, pxz, vz, vx, lamb, muon, dt, coef, kk,
                  ax, az, bx, bz, kxc, kzc,
                  pml_vxx, pml_vzz, pml_vxz, pml_vzx, tau_p, tau_s,
                  xx, zz, xz, zx):
    """
    CUDA kernel to compute spatial derivatives for stress update.
    
    This kernel computes the finite-difference approximations for the spatial
    derivatives of the velocity fields. It calculates intermediate arrays xx, zz, xz,
    and zx that are later used to update the stress fields.
    
    Parameters:
      shape: Tuple (pnz, pnx) representing the total grid dimensions (including boundaries).
      dx, dz: Spatial grid spacings in x and z directions.
      pxx, pzz, pxz: Stress field arrays that will be updated.
      vz, vx: Velocity field arrays (vertical and horizontal components).
      lamb, muon: Elastic parameters (lambda and shear modulus multiplied by density).
      dt: Time step size.
      coef: Array of finite-difference coefficients.
      kk: Order (or half-width) of the finite-difference stencil.
      ax, az, bx, bz, kxc, kzc: CPML (absorbing boundary) parameters arrays.
      pml_vxx, pml_vzz, pml_vxz, pml_vzx: Arrays to store the CPML correction terms.
      tau_p, tau_s: Arrays for wavefield separation (P and S components).
      xx, zz, xz, zx: Intermediate arrays to hold spatial derivative results.
    """
    # Extract total grid dimensions (including boundaries)
    pnx = shape[1]
    pnz = shape[0]

    # Determine the current thread's indices in the computational grid
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Only process interior grid points where a full finite-difference stencil is available
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:
        # Initialize temporary accumulators for finite-difference calculation
        diff1, diff2 = 0.0, 0.0

        # Compute spatial derivative components xx and zz:
        # - xx: Finite-difference approximation of derivative of vx in x direction.
        # - zz: Finite-difference approximation of derivative of vz in z direction.
        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (vx[iz, ix + ii] - vx[iz, ix - ii + 1])
            diff2 += coef[ii - 1] * (vz[iz + ii - 1, ix] - vz[iz - ii, ix])
        xx[iz, ix] = diff1 / dx
        zz[iz, ix] = diff2 / dz

        # Reset accumulators and compute cross derivatives xz and zx:
        # - xz: Finite-difference approximation related to the derivative of vx in the z direction.
        # - zx: Finite-difference approximation related to the derivative of vz in the x direction.
        diff1, diff2 = 0.0, 0.0
        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (vx[iz + ii, ix] - vx[iz - ii + 1, ix])
            diff2 += coef[ii - 1] * (vz[iz, ix + ii - 1] - vz[iz, ix - ii])
        xz[iz, ix] = diff1 / dz
        zx[iz, ix] = diff2 / dx

@cuda.jit
def updatep2_cuda(shape, dx, dz, pxx, pzz, pxz, vz, vx, lamb, muon, dt, coef, kk,
                  ax, az, bx, bz, kxc, kzc,
                  pml_vxx, pml_vzz, pml_vxz, pml_vzx, tau_p, tau_s,
                  xx, zz, xz, zx):
    """
    CUDA kernel to update stress fields using CPML corrections.
    
    This kernel first updates the CPML (absorbing boundary) correction terms by
    blending the computed spatial derivatives with the CPML parameters. It then
    updates the stress fields using the elastic parameters and the corrected spatial derivatives.
    Finally, it updates the separated wavefield components (tau_p and tau_s) used for further processing.
    
    Parameters:
      shape: Tuple (pnz, pnx) representing the grid dimensions (including boundaries).
      dx, dz: Spatial grid spacings in x and z directions.
      pxx, pzz, pxz: Stress field arrays.
      vz, vx: Velocity field arrays.
      lamb, muon: Elastic parameters (lambda and shear modulus multiplied by density).
      dt: Time step size.
      coef: Finite-difference coefficients array.
      kk: Order (or half-width) of the finite-difference stencil.
      ax, az, bx, bz, kxc, kzc: CPML parameters arrays.
      pml_vxx, pml_vzz, pml_vxz, pml_vzx: Arrays storing the CPML correction terms.
      tau_p, tau_s: Arrays for wavefield separation (for P and S components).
      xx, zz, xz, zx: Intermediate derivative arrays computed in updatep1_cuda.
    """
    # Extract total grid dimensions (including CPML boundaries)
    pnx = shape[1]
    pnz = shape[0]

    # Calculate the global indices of the current thread
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Process only interior grid points that can accommodate the full finite-difference stencil
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:
        # -------------------------------------------------------------
        # Update CPML correction terms for each spatial derivative
        # -------------------------------------------------------------
        # For the xx component:
        pml_vxx[iz, ix] = bx[iz, ix] * pml_vxx[iz, ix] + ax[iz, ix] * xx[iz, ix]
        xx[iz, ix] = xx[iz, ix] / kxc[iz, ix] + pml_vxx[iz, ix]

        # For the zz component:
        pml_vzz[iz, ix] = bz[iz, ix] * pml_vzz[iz, ix] + az[iz, ix] * zz[iz, ix]
        zz[iz, ix] = zz[iz, ix] / kzc[iz, ix] + pml_vzz[iz, ix]

        # For the xz component:
        pml_vxz[iz, ix] = bz[iz, ix] * pml_vxz[iz, ix] + az[iz, ix] * xz[iz, ix]
        xz[iz, ix] = xz[iz, ix] / kzc[iz, ix] + pml_vxz[iz, ix]

        # For the zx component:
        pml_vzx[iz, ix] = bx[iz, ix] * pml_vzx[iz, ix] + ax[iz, ix] * zx[iz, ix]
        zx[iz, ix] = zx[iz, ix] / kxc[iz, ix] + pml_vzx[iz, ix]

        # -------------------------------------------------------------
        # Update the stress fields using the corrected spatial derivatives
        # -------------------------------------------------------------
        # pxx: Stress component in the x direction, updated using both xx and zz derivatives
        pxx[iz, ix] += dt * ((lamb[iz, ix] + 2 * muon[iz, ix]) * xx[iz, ix] + lamb[iz, ix] * zz[iz, ix])
        # pzz: Stress component in the z direction, similarly updated
        pzz[iz, ix] += dt * ((lamb[iz, ix] + 2 * muon[iz, ix]) * zz[iz, ix] + lamb[iz, ix] * xx[iz, ix])
        # pxz: Off-diagonal stress component, updated using the sum of xz and zx components
        pxz[iz, ix] += dt * muon[iz, ix] * (xz[iz, ix] + zx[iz, ix])

        # -------------------------------------------------------------
        # Separate the wavefield into its P-wave and S-wave components
        # -------------------------------------------------------------
        # tau_p accumulates the compressional (P-wave) component based on the divergence (xx + zz)
        tau_p[iz, ix] += dt * ((lamb[iz, ix] + 2 * muon[iz, ix]) * (xx[iz, ix] + zz[iz, ix]))
        # tau_s accumulates the shear (S-wave) component based on the difference between cross derivatives
        tau_s[iz, ix] += dt * (muon[iz, ix] * (-zx[iz, ix] + xz[iz, ix]))
