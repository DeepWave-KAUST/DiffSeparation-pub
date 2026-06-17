import numpy as np
from numba import cuda

@cuda.jit
def calmid1_cuda(shape, dx, dz, coef, kk, vx, vz, xx, zz, xz, zx, divw, curw):
    """
    CUDA kernel to compute intermediate spatial derivatives from the velocity fields.
    
    This kernel calculates the divergence (divw) and curl (curw) components of the wavefield,
    as well as intermediate derivatives (xx, zz, xz, zx) used in further processing.
    
    Parameters:
      shape: Tuple (pnz, pnx) giving the total grid dimensions including boundaries.
      dx, dz: Spatial grid spacing in x and z directions.
      coef: Array of finite-difference coefficients.
      kk: Order of the finite-difference scheme.
      vx, vz: Input velocity fields in the x (horizontal) and z (vertical) directions.
      xx, zz: Output arrays to store computed spatial derivatives along x and z, respectively.
      xz, zx: Output arrays to store additional cross-derivative components.
      divw: Output array to store the divergence (sum of xx and zz).
      curw: Output array to store the curl (difference between xz and zx).
    """
    # Extract grid dimensions from the shape parameter
    pnx = shape[1]
    pnz = shape[0]

    # Determine the current thread's indices in the grid
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Check if the thread is within the valid computation region (avoid boundaries)
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:
        # Initialize temporary variables for spatial derivative computations.
        # Compute derivatives in the x and z directions for divergence.
        diff1, diff2 = 0.0, 0.0
        for ii in range(1, kk + 1):
            # Finite-difference approximation for x-derivative of vx:
            # The difference between forward and backward samples weighted by coefficients.
            diff1 += coef[ii - 1] * (vx[iz, ix + ii] - vx[iz, ix - ii + 1])
            # Finite-difference approximation for z-derivative of vz:
            diff2 += coef[ii - 1] * (vz[iz + ii - 1, ix] - vz[iz - ii, ix])
        # Normalize the derivative estimates by the grid spacing.
        xx[iz, ix] = diff1 / dx
        zz[iz, ix] = diff2 / dz
        # The divergence is the sum of the spatial derivatives along x and z.
        divw[iz, ix] = xx[iz, ix] + zz[iz, ix]

        # Next, compute cross-derivative components related to the curl.
        diff1, diff2 = 0.0, 0.0
        for ii in range(1, kk + 1):
            # Finite-difference approximation for x-derivative of vz (shifted index)
            diff1 += coef[ii - 1] * (vz[iz, ix + ii - 1] - vz[iz, ix - ii])
            # Finite-difference approximation for z-derivative of vx (shifted index)
            diff2 += coef[ii - 1] * (vx[iz + ii, ix] - vx[iz - ii + 1, ix])
        # Normalize the derivative estimates by the grid spacing.
        zx[iz, ix] = diff1 / dx
        xz[iz, ix] = diff2 / dz
        # The curl (curw) is defined as the difference between these cross-derivatives.
        curw[iz, ix] = xz[iz, ix] - zx[iz, ix]

@cuda.jit
def calmid2_cuda(shape, dx, dz, coef, kk, vx, vz, divw, curw, 
                 mid_vxp, mid_vzp, mid_vxs, mid_vzs):
    """
    CUDA kernel to compute additional intermediate variables used for elastic wave separation.
    
    This kernel calculates derivatives based on the divergence (divw) and curl (curw) 
    computed in the first kernel, which are used to separate the wavefield into different components.
    
    Parameters:
      shape: Tuple (pnz, pnx) giving the total grid dimensions including boundaries.
      dx, dz: Spatial grid spacing in x and z directions.
      coef: Array of finite-difference coefficients.
      kk: Order of the finite-difference scheme.
      vx, vz: Input velocity fields (not used directly in computation here but passed for consistency).
      divw: Array containing the divergence of the wavefield computed previously.
      curw: Array containing the curl of the wavefield computed previously.
      mid_vxp, mid_vzp: Output arrays to store the derivatives (tau_px and tau_pz) from divergence.
      mid_vxs, mid_vzs: Output arrays to store the derivatives (tau_sx and tau_sz) from curl.
    """
    # Extract grid dimensions from the shape parameter
    pnx = shape[1]
    pnz = shape[0]

    # Determine the current thread's indices in the grid
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Check if the thread is within the valid computation region (avoid boundaries)
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:
        # Compute derivatives from the divergence (divw) for tau_px and tau_pz
        diff1, diff2 = 0.0, 0.0
        for ii in range(1, kk + 1):
            # Finite-difference approximation along x using divergence values
            diff1 += coef[ii - 1] * (divw[iz, ix + ii - 1] - divw[iz, ix - ii])
            # Finite-difference approximation along z using divergence values
            diff2 += coef[ii - 1] * (divw[iz + ii, ix] - divw[iz - ii + 1, ix])
        # Normalize by grid spacing to obtain the derivative estimates.
        mid_vxp[iz, ix] = diff1 / dx
        mid_vzp[iz, ix] = diff2 / dz

        # Compute derivatives from the curl (curw) for tau_sx and tau_sz
        diff1, diff2 = 0.0, 0.0
        for ii in range(1, kk + 1):
            # Finite-difference approximation along z using curl values
            diff1 += coef[ii - 1] * (curw[iz + ii - 1, ix] - curw[iz - ii, ix])
            # Finite-difference approximation along x using curl values
            diff2 += coef[ii - 1] * (curw[iz, ix + ii] - curw[iz, ix - ii + 1])
        # Normalize by grid spacing; note the negative sign for the x-derivative of curl.
        mid_vxs[iz, ix] = diff1 / dz
        mid_vzs[iz, ix] = -diff2 / dx
