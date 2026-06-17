import numpy as np
from numba import cuda

@cuda.jit
def updatev1_cuda(shape, dx, dz, vz, vx, pxx, pzz, pxz, rho, dt, coef, kk,
                  ax, az, bx, bz, kxc, kzc,
                  pml_pxxx, pml_pxzz, pml_pxzx, pml_pzzz, tau_p, s_px, s_pz,
                  pml_tau_px, pml_tau_pz, tau_s, s_sx, s_sz,
                  pml_tau_sx, pml_tau_sz, 
                  xxx, xzz, xzx, zzz, tau_px, tau_pz, tau_sx, tau_sz):
    """
    CUDA kernel to compute spatial derivatives of the stress field for the velocity update.
    
    This kernel calculates intermediate derivative fields based on the stress components 
    (pxx, pzz, pxz) and wavefield separation variables (tau_p, tau_s) using a finite-difference 
    scheme of order 'kk' with coefficients 'coef'. These intermediate arrays (xxx, xzz, xzx, zzz,
    tau_px, tau_pz, tau_sx, tau_sz) are used in the next kernel for updating the velocity fields.
    
    Parameters:
      shape: Tuple (pnz, pnx) representing the grid dimensions (including boundary layers).
      dx, dz: Spatial grid spacings in x and z directions.
      vz, vx: Vertical and horizontal velocity fields.
      pxx, pzz, pxz: Stress field components.
      rho: Density field.
      dt: Time step size.
      coef: Finite-difference coefficients.
      kk: Order (or half-width) of the finite-difference stencil.
      ax, az, bx, bz, kxc, kzc: CPML (absorbing boundary) parameters.
      pml_pxxx, pml_pxzz, pml_pxzx, pml_pzzz: CPML correction arrays for stress derivatives.
      tau_p, tau_s: Wavefield separation variables (related to P-wave and S-wave components).
      s_px, s_pz, s_sx, s_sz: Separated wavefield arrays.
      pml_tau_px, pml_tau_pz, pml_tau_sx, pml_tau_sz: CPML correction arrays for separated fields.
      xxx, xzz, xzx, zzz: Intermediate arrays for spatial derivatives of stress.
      tau_px, tau_pz, tau_sx, tau_sz: Intermediate arrays for spatial derivatives of wavefield separation variables.
    """
    # Get the full grid dimensions (including boundaries)
    pnx = shape[1]
    pnz = shape[0]

    # Calculate the global indices for the current CUDA thread
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Ensure the current thread is in the interior region where the stencil is fully available
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:
        # ------------------------------------------
        # Compute first set of spatial derivatives (xxx and xzz)
        # ------------------------------------------
        diff1, diff2 = 0.0, 0.0
        # Loop over the stencil width
        for ii in range(1, kk + 1):
            # Approximate the x-derivative of pxx using finite differences
            diff1 += coef[ii - 1] * (pxx[iz, ix + ii - 1] - pxx[iz, ix - ii])
            # Approximate the z-derivative of pxz using finite differences
            diff2 += coef[ii - 1] * (pxz[iz + ii - 1, ix] - pxz[iz - ii + 1, ix])
        # Normalize by grid spacing to get derivative estimates
        xxx[iz, ix] = diff1 / dx
        xzz[iz, ix] = diff2 / dz

        # ------------------------------------------
        # Compute second set of spatial derivatives (xzx and zzz)
        # ------------------------------------------
        diff1 = 0.0
        diff2 = 0.0
        for ii in range(1, kk + 1):
            # Approximate the x-derivative of pxz using finite differences
            diff1 += coef[ii - 1] * (pxz[iz, ix + ii] - pxz[iz, ix - ii + 1])
            # Approximate the z-derivative of pzz using finite differences
            diff2 += coef[ii - 1] * (pzz[iz + ii, ix] - pzz[iz - ii + 1, ix])
        xzx[iz, ix] = diff1 / dx
        zzz[iz, ix] = diff2 / dz

        # ------------------------------------------
        # Compute spatial derivatives of the wavefield separation (tau_px and tau_pz)
        # ------------------------------------------
        diff1, diff2 = 0.0, 0.0
        for ii in range(1, kk + 1):
            # Derivative along x for tau_p
            diff1 += coef[ii - 1] * (tau_p[iz, ix + ii - 1] - tau_p[iz, ix - ii])
            # Derivative along z for tau_p
            diff2 += coef[ii - 1] * (tau_p[iz + ii, ix] - tau_p[iz - ii + 1, ix])
        tau_px[iz, ix] = diff1 / dx
        tau_pz[iz, ix] = diff2 / dz

        # ------------------------------------------
        # Compute spatial derivatives of the wavefield separation (tau_sx and tau_sz)
        # ------------------------------------------
        diff1 = 0.0
        diff2 = 0.0
        for ii in range(1, kk + 1):
            # Derivative along z for tau_s
            diff1 += coef[ii - 1] * (tau_s[iz + ii - 1, ix] - tau_s[iz - ii, ix])
            # Derivative along x for tau_s
            diff2 += coef[ii - 1] * (tau_s[iz, ix + ii] - tau_s[iz, ix - ii + 1])
        # Note the negative sign for tau_sz to account for the derivative direction
        tau_sx[iz, ix] = diff1 / dz
        tau_sz[iz, ix] = -diff2 / dx

@cuda.jit
def updatev2_cuda(shape, dx, dz, vz, vx, pxx, pzz, pxz, rho, dt, coef, kk,
                  ax, az, bx, bz, kxc, kzc,
                  pml_pxxx, pml_pxzz, pml_pxzx, pml_pzzz, tau_p, s_px, s_pz,
                  pml_tau_px, pml_tau_pz, tau_s, s_sx, s_sz,
                  pml_tau_sx, pml_tau_sz, 
                  xxx, xzz, xzx, zzz, tau_px, tau_pz, tau_sx, tau_sz):
    """
    CUDA kernel to update the velocity fields using the previously computed spatial derivatives.
    
    This kernel first updates the CPML (absorbing boundary) correction terms for the spatial
    derivative arrays (xxx, xzz, xzx, zzz, tau_px, tau_pz, tau_sx, tau_sz) by blending them with
    the CPML parameters. Then, it updates the velocity fields (vx, vz) based on these corrected derivatives.
    Finally, it updates the separated wavefield components (s_px, s_pz, s_sx, s_sz) using the corrected 
    derivatives of the separated fields (tau_px, tau_pz, tau_sx, tau_sz).
    
    Parameters:
      shape: Tuple (pnz, pnx) representing the grid dimensions (including boundaries).
      dx, dz: Spatial grid spacings in x and z directions.
      vz, vx: Vertical and horizontal velocity fields.
      pxx, pzz, pxz: Stress field components.
      rho: Density field.
      dt: Time step size.
      coef: Finite-difference coefficients.
      kk: Order of the finite-difference stencil.
      ax, az, bx, bz, kxc, kzc: CPML parameter arrays.
      pml_pxxx, pml_pxzz, pml_pxzx, pml_pzzz: CPML correction arrays for stress derivatives.
      tau_p, tau_s: Wavefield separation variables.
      s_px, s_pz, s_sx, s_sz: Separated wavefield arrays.
      pml_tau_px, pml_tau_pz, pml_tau_sx, pml_tau_sz: CPML correction arrays for separated fields.
      xxx, xzz, xzx, zzz: Intermediate arrays for spatial derivatives computed in updatev1_cuda.
      tau_px, tau_pz, tau_sx, tau_sz: Intermediate arrays for spatial derivatives of wavefield separation.
    """
    # Get grid dimensions (including CPML boundaries)
    pnx = shape[1]
    pnz = shape[0]

    # Calculate global indices for the current thread
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Process only interior grid points where the full finite-difference stencil is available
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:
        # ------------------------------------------------------
        # Update CPML correction terms for spatial derivative arrays
        # ------------------------------------------------------
        # Update for the xxx component using CPML damping parameters
        pml_pxxx[iz, ix] = bx[iz, ix] * pml_pxxx[iz, ix] + ax[iz, ix] * xxx[iz, ix]
        xxx[iz, ix] = xxx[iz, ix] / kxc[iz, ix] + pml_pxxx[iz, ix]

        # Update for the xzz component
        pml_pxzz[iz, ix] = bz[iz, ix] * pml_pxzz[iz, ix] + az[iz, ix] * xzz[iz, ix]
        xzz[iz, ix] = xzz[iz, ix] / kzc[iz, ix] + pml_pxzz[iz, ix]

        # Update for the xzx component
        pml_pxzx[iz, ix] = bx[iz, ix] * pml_pxzx[iz, ix] + ax[iz, ix] * xzx[iz, ix]
        xzx[iz, ix] = xzx[iz, ix] / kxc[iz, ix] + pml_pxzx[iz, ix]

        # Update for the zzz component
        pml_pzzz[iz, ix] = bz[iz, ix] * pml_pzzz[iz, ix] + az[iz, ix] * zzz[iz, ix]
        zzz[iz, ix] = zzz[iz, ix] / kzc[iz, ix] + pml_pzzz[iz, ix]

        # Update CPML correction for separated field derivatives (tau_px and tau_pz)
        pml_tau_px[iz, ix] = bx[iz, ix] * pml_tau_px[iz, ix] + ax[iz, ix] * tau_px[iz, ix]
        tau_px[iz, ix] = tau_px[iz, ix] / kxc[iz, ix] + pml_tau_px[iz, ix]

        pml_tau_pz[iz, ix] = bz[iz, ix] * pml_tau_pz[iz, ix] + az[iz, ix] * tau_pz[iz, ix]
        tau_pz[iz, ix] = tau_pz[iz, ix] / kzc[iz, ix] + pml_tau_pz[iz, ix]

        # Update CPML correction for separated field derivatives (tau_sx and tau_sz)
        pml_tau_sx[iz, ix] = bx[iz, ix] * pml_tau_sx[iz, ix] + ax[iz, ix] * tau_sx[iz, ix]
        tau_sx[iz, ix] = tau_sx[iz, ix] / kxc[iz, ix] + pml_tau_sx[iz, ix]

        pml_tau_sz[iz, ix] = bz[iz, ix] * pml_tau_sz[iz, ix] + az[iz, ix] * tau_sz[iz, ix]
        tau_sz[iz, ix] = tau_sz[iz, ix] / kzc[iz, ix] + pml_tau_sz[iz, ix]

        # ------------------------------------------------------
        # Update velocity fields using the corrected derivatives
        # ------------------------------------------------------
        # Update horizontal velocity (vx) using the sum of the derivatives in the x direction
        vx[iz, ix] += dt * (xxx[iz, ix] + xzz[iz, ix]) / rho[iz, ix]
        # Update vertical velocity (vz) using the sum of the derivatives in the z direction
        vz[iz, ix] += dt * (xzx[iz, ix] + zzz[iz, ix]) / rho[iz, ix]

        # ------------------------------------------------------
        # Update the separated wavefields (for further analysis)
        # ------------------------------------------------------
        s_px[iz, ix] += dt * tau_px[iz, ix] / rho[iz, ix]
        s_pz[iz, ix] += dt * tau_pz[iz, ix] / rho[iz, ix]
        s_sx[iz, ix] += dt * tau_sx[iz, ix] / rho[iz, ix]
        s_sz[iz, ix] += dt * tau_sz[iz, ix] / rho[iz, ix]
