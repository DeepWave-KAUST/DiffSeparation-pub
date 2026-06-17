import numpy as np
from numba import cuda


# ================================================================
# CUDA kernel: calmid1_cuda
# ================================================================
# Purpose:
#   This CUDA kernel computes the first-order spatial derivatives of the
#   two-component elastic velocity wavefield (vx, vz), and then obtains:
#
#       1. divw = ∂vx/∂x + ∂vz/∂z
#          which is the divergence of the velocity wavefield.
#
#       2. curw = ∂vx/∂z - ∂vz/∂x
#          which is the curl-related quantity of the velocity wavefield.
#
#   These two quantities are intermediate variables for Helmholtz
#   decomposition-based P/S wavefield separation.
#
# Inputs:
#   shape : tuple
#       Shape of the padded computational domain, including CPML layers.
#       shape[0] is the number of grid points in z direction.
#       shape[1] is the number of grid points in x direction.
#
#   dx, dz : float
#       Spatial grid intervals in x and z directions.
#
#   coef : device array
#       Finite-difference coefficients used for high-order spatial derivatives.
#
#   kk : int
#       Number of finite-difference coefficients. It also defines the half-width
#       of the finite-difference stencil.
#
#   vx, vz : device arrays
#       Horizontal and vertical components of the elastic velocity wavefield.
#
# Outputs:
#   xx : device array
#       Approximation of ∂vx/∂x.
#
#   zz : device array
#       Approximation of ∂vz/∂z.
#
#   xz : device array
#       Approximation of ∂vx/∂z.
#
#   zx : device array
#       Approximation of ∂vz/∂x.
#
#   divw : device array
#       Divergence of the velocity wavefield.
#
#   curw : device array
#       Curl-related quantity of the velocity wavefield.
# ================================================================
@cuda.jit
def calmid1_cuda(shape, dx, dz, coef, kk, vx, vz, xx, zz, xz, zx, divw, curw):

    # Number of grid points in x and z directions for the padded domain.
    pnx = shape[1]
    pnz = shape[0]

    # Global thread indices.
    # Here, iz corresponds to the z-direction index,
    # and ix corresponds to the x-direction index.
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Avoid computing near the boundary where the finite-difference stencil
    # would exceed the valid computational domain.
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:

        # ------------------------------------------------------------
        # Compute ∂vx/∂x and ∂vz/∂z.
        #
        # diff1 approximates the x-derivative of vx.
        # diff2 approximates the z-derivative of vz.
        #
        # The finite-difference stencil is written in a staggered-grid-like
        # form, which is commonly used in elastic wave equation modeling.
        # ------------------------------------------------------------
        diff1, diff2 = 0.0, 0.0

        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (
                vx[iz, ix + ii] - vx[iz, ix - ii + 1]
            )

            diff2 += coef[ii - 1] * (
                vz[iz + ii - 1, ix] - vz[iz - ii, ix]
            )

        # Store the spatial derivatives.
        xx[iz, ix] = diff1 / dx
        zz[iz, ix] = diff2 / dz

        # Compute divergence:
        # div(v) = ∂vx/∂x + ∂vz/∂z
        divw[iz, ix] = xx[iz, ix] + zz[iz, ix]

        # ------------------------------------------------------------
        # Compute ∂vz/∂x and ∂vx/∂z.
        #
        # diff1 approximates the x-derivative of vz.
        # diff2 approximates the z-derivative of vx.
        # ------------------------------------------------------------
        diff1, diff2 = 0.0, 0.0

        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (
                vz[iz, ix + ii - 1] - vz[iz, ix - ii]
            )

            diff2 += coef[ii - 1] * (
                vx[iz + ii, ix] - vx[iz - ii + 1, ix]
            )

        # Store the spatial derivatives.
        zx[iz, ix] = diff1 / dx
        xz[iz, ix] = diff2 / dz

        # Compute curl-related quantity:
        # curl(v) = ∂vx/∂z - ∂vz/∂x
        curw[iz, ix] = xz[iz, ix] - zx[iz, ix]


# ================================================================
# CUDA kernel: calmid2_cuda
# ================================================================
# Purpose:
#   This CUDA kernel computes the spatial derivatives of the divergence
#   and curl-related quantities obtained from calmid1_cuda.
#
#   Specifically, it computes:
#
#       mid_vxp = ∂(divw)/∂x
#       mid_vzp = ∂(divw)/∂z
#
#   which are associated with the P-wave component, and
#
#       mid_vxs = ∂(curw)/∂z
#       mid_vzs = -∂(curw)/∂x
#
#   which are associated with the S-wave component.
#
#   These quantities are intermediate variables used for elastic
#   wavefield separation based on Helmholtz decomposition.
#
# Inputs:
#   shape : tuple
#       Shape of the padded computational domain, including CPML layers.
#
#   dx, dz : float
#       Spatial grid intervals in x and z directions.
#
#   coef : device array
#       Finite-difference coefficients.
#
#   kk : int
#       Number of finite-difference coefficients.
#
#   vx, vz : device arrays
#       Horizontal and vertical velocity components.
#       These variables are included for consistency with the calling interface,
#       although they are not directly used in this kernel.
#
#   divw : device array
#       Divergence of the velocity wavefield.
#
#   curw : device array
#       Curl-related quantity of the velocity wavefield.
#
# Outputs:
#   mid_vxp, mid_vzp : device arrays
#       Intermediate P-wave-related components computed from the gradient
#       of the divergence field.
#
#   mid_vxs, mid_vzs : device arrays
#       Intermediate S-wave-related components computed from the curl field.
# ================================================================
@cuda.jit
def calmid2_cuda(shape, dx, dz, coef, kk, vx, vz, divw, curw,
                 mid_vxp, mid_vzp, mid_vxs, mid_vzs):

    # Number of grid points in x and z directions for the padded domain.
    pnx = shape[1]
    pnz = shape[0]

    # Global thread indices.
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Avoid computing near the boundary where the finite-difference stencil
    # would exceed the valid computational domain.
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:

        # ------------------------------------------------------------
        # Compute the gradient of the divergence field.
        #
        # mid_vxp approximates ∂(divw)/∂x.
        # mid_vzp approximates ∂(divw)/∂z.
        #
        # These two components are related to the P-wave part of the
        # elastic wavefield.
        # ------------------------------------------------------------
        diff1, diff2 = 0.0, 0.0

        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (
                divw[iz, ix + ii - 1] - divw[iz, ix - ii]
            )

            diff2 += coef[ii - 1] * (
                divw[iz + ii, ix] - divw[iz - ii + 1, ix]
            )

        mid_vxp[iz, ix] = diff1 / dx
        mid_vzp[iz, ix] = diff2 / dz

        # ------------------------------------------------------------
        # Compute derivatives of the curl-related field.
        #
        # mid_vxs approximates ∂(curw)/∂z.
        # mid_vzs approximates -∂(curw)/∂x.
        #
        # These two components are related to the S-wave part of the
        # elastic wavefield.
        # ------------------------------------------------------------
        diff1, diff2 = 0.0, 0.0

        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (
                curw[iz + ii - 1, ix] - curw[iz - ii, ix]
            )

            diff2 += coef[ii - 1] * (
                curw[iz, ix + ii] - curw[iz, ix - ii + 1]
            )

        mid_vxs[iz, ix] = diff1 / dz
        mid_vzs[iz, ix] = -diff2 / dx