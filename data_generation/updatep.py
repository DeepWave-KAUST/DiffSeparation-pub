import numpy as np
from numba import cuda


# ================================================================
# CUDA kernel: updatep1_cuda
# ================================================================
# Purpose:
#   This CUDA kernel computes the spatial derivatives of the elastic
#   particle velocity components vx and vz.
#
#   Specifically, it computes:
#
#       xx = ∂vx/∂x
#       zz = ∂vz/∂z
#       xz = ∂vx/∂z
#       zx = ∂vz/∂x
#
#   These derivatives are later used in updatep2_cuda to update the
#   elastic stress components:
#
#       pxx, pzz, pxz
#
#   The same derivatives are also used to update tau_p and tau_s,
#   which are auxiliary stress-like fields used for P/S wavefield
#   separation.
#
# Inputs:
#   shape : tuple
#       Shape of the padded computational domain, including CPML layers.
#       shape[0] is the number of grid points in z direction.
#       shape[1] is the number of grid points in x direction.
#
#   dx, dz : float
#       Spatial grid intervals in the x and z directions.
#
#   pxx, pzz, pxz : device arrays
#       Elastic stress components. They are passed for interface consistency,
#       but are not directly updated in this first kernel.
#
#   vz, vx : device arrays
#       Vertical and horizontal particle velocity components.
#
#   lamb : device array
#       Lamé parameter lambda.
#
#   muon : device array
#       Shear modulus mu.
#
#   dt : float
#       Time sampling interval.
#
#   coef : device array
#       Finite-difference coefficients for high-order spatial derivatives.
#
#   kk : int
#       Number of finite-difference coefficients. It also controls the
#       half-width of the finite-difference stencil.
#
#   ax, az, bx, bz, kxc, kzc :
#       CPML coefficients. They are passed here for interface consistency,
#       but are mainly used in updatep2_cuda.
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
# ================================================================
@cuda.jit
def updatep1_cuda(shape, dx, dz, pxx, pzz, pxz, vz, vx, lamb, muon, dt, coef, kk,
                 ax, az, bx, bz, kxc, kzc,
                 pml_vxx, pml_vzz, pml_vxz, pml_vzx, tau_p, tau_s,
                 xx, zz, xz, zx):

    # Number of grid points in x and z directions for the padded domain.
    pnx = shape[1]
    pnz = shape[0]

    # Global thread indices.
    # iz corresponds to the z-direction index.
    # ix corresponds to the x-direction index.
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Avoid computing near the boundary where the finite-difference stencil
    # would exceed the valid computational domain.
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:

        # ------------------------------------------------------------
        # Compute the normal velocity derivatives:
        #
        #   xx = ∂vx/∂x
        #   zz = ∂vz/∂z
        #
        # These two terms are used to update the normal stress components:
        #
        #   pxx and pzz
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

        xx[iz, ix] = diff1 / dx
        zz[iz, ix] = diff2 / dz

        # ------------------------------------------------------------
        # Compute the shear-related velocity derivatives:
        #
        #   xz = ∂vx/∂z
        #   zx = ∂vz/∂x
        #
        # These two terms are used to update the shear stress component:
        #
        #   pxz
        # ------------------------------------------------------------
        diff1, diff2 = 0.0, 0.0

        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (
                vx[iz + ii, ix] - vx[iz - ii + 1, ix]
            )

            diff2 += coef[ii - 1] * (
                vz[iz, ix + ii - 1] - vz[iz, ix - ii]
            )

        xz[iz, ix] = diff1 / dz
        zx[iz, ix] = diff2 / dx


# ================================================================
# CUDA kernel: updatep2_cuda
# ================================================================
# Purpose:
#   This CUDA kernel applies CPML corrections to the velocity-derivative
#   terms computed by updatep1_cuda, and then updates the elastic stress
#   fields:
#
#       pxx, pzz, pxz
#
#   It also updates two auxiliary stress-like fields:
#
#       tau_p and tau_s
#
#   where tau_p is related to the compressional/P-wave component and
#   tau_s is related to the shear/S-wave component.
#
#   In the main time-stepping loop, tau_p and tau_s are subsequently used
#   in updatev1_cuda and updatev2_cuda to obtain separated P- and S-wave
#   velocity components.
#
# Inputs:
#   xx, zz, xz, zx : device arrays
#       Velocity spatial derivatives computed by updatep1_cuda.
#
#   lamb : device array
#       Lamé parameter lambda.
#
#   muon : device array
#       Shear modulus mu.
#
#   dt : float
#       Time sampling interval.
#
#   ax, az, bx, bz, kxc, kzc : device arrays
#       CPML damping and scaling coefficients.
#
#   pml_vxx, pml_vzz, pml_vxz, pml_vzx : device arrays
#       CPML memory variables corresponding to the velocity derivatives.
#
# Outputs:
#   pxx, pzz, pxz : device arrays
#       Updated elastic stress components.
#
#   tau_p : device array
#       Updated P-wave-related auxiliary stress field.
#
#   tau_s : device array
#       Updated S-wave-related auxiliary stress field.
# ================================================================
@cuda.jit
def updatep2_cuda(shape, dx, dz, pxx, pzz, pxz, vz, vx, lamb, muon, dt, coef, kk,
                 ax, az, bx, bz, kxc, kzc,
                 pml_vxx, pml_vzz, pml_vxz, pml_vzx, tau_p, tau_s,
                 xx, zz, xz, zx):

    # Number of grid points in x and z directions for the padded domain.
    pnx = shape[1]
    pnz = shape[0]

    # Global thread indices.
    # iz corresponds to the z-direction index.
    # ix corresponds to the x-direction index.
    iz = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    ix = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # Avoid computing near the boundary where the finite-difference stencil
    # would exceed the valid computational domain.
    if kk <= iz < pnz - kk and kk <= ix < pnx - kk:

        # ------------------------------------------------------------
        # Apply CPML correction to xx = ∂vx/∂x.
        #
        # pml_vxx is the CPML memory variable for this derivative.
        # kxc is the CPML scaling factor in the x direction.
        # ------------------------------------------------------------
        pml_vxx[iz, ix] = (
            bx[iz, ix] * pml_vxx[iz, ix]
            + ax[iz, ix] * xx[iz, ix]
        )
        xx[iz, ix] = xx[iz, ix] / kxc[iz, ix] + pml_vxx[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to zz = ∂vz/∂z.
        #
        # pml_vzz is the CPML memory variable for this derivative.
        # kzc is the CPML scaling factor in the z direction.
        # ------------------------------------------------------------
        pml_vzz[iz, ix] = (
            bz[iz, ix] * pml_vzz[iz, ix]
            + az[iz, ix] * zz[iz, ix]
        )
        zz[iz, ix] = zz[iz, ix] / kzc[iz, ix] + pml_vzz[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to xz = ∂vx/∂z.
        #
        # pml_vxz is the CPML memory variable for this derivative.
        # Since xz is a z-direction derivative, z-direction CPML
        # coefficients are used.
        # ------------------------------------------------------------
        pml_vxz[iz, ix] = (
            bz[iz, ix] * pml_vxz[iz, ix]
            + az[iz, ix] * xz[iz, ix]
        )
        xz[iz, ix] = xz[iz, ix] / kzc[iz, ix] + pml_vxz[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to zx = ∂vz/∂x.
        #
        # pml_vzx is the CPML memory variable for this derivative.
        # Since zx is an x-direction derivative, x-direction CPML
        # coefficients are used.
        # ------------------------------------------------------------
        pml_vzx[iz, ix] = (
            bx[iz, ix] * pml_vzx[iz, ix]
            + ax[iz, ix] * zx[iz, ix]
        )
        zx[iz, ix] = zx[iz, ix] / kxc[iz, ix] + pml_vzx[iz, ix]

        # ------------------------------------------------------------
        # Update normal stress component pxx.
        #
        # For isotropic elastic media:
        #
        #   pxx += dt * [ (lambda + 2 * mu) * ∂vx/∂x
        #                 + lambda * ∂vz/∂z ]
        #
        # Here:
        #   xx = ∂vx/∂x
        #   zz = ∂vz/∂z
        # ------------------------------------------------------------
        pxx[iz, ix] += dt * (
            (lamb[iz, ix] + 2 * muon[iz, ix]) * xx[iz, ix]
            + lamb[iz, ix] * zz[iz, ix]
        )

        # ------------------------------------------------------------
        # Update normal stress component pzz.
        #
        #   pzz += dt * [ (lambda + 2 * mu) * ∂vz/∂z
        #                 + lambda * ∂vx/∂x ]
        # ------------------------------------------------------------
        pzz[iz, ix] += dt * (
            (lamb[iz, ix] + 2 * muon[iz, ix]) * zz[iz, ix]
            + lamb[iz, ix] * xx[iz, ix]
        )

        # ------------------------------------------------------------
        # Update shear stress component pxz.
        #
        #   pxz += dt * mu * (∂vx/∂z + ∂vz/∂x)
        #
        # Here:
        #   xz = ∂vx/∂z
        #   zx = ∂vz/∂x
        # ------------------------------------------------------------
        pxz[iz, ix] += dt * muon[iz, ix] * (
            xz[iz, ix] + zx[iz, ix]
        )

        # ------------------------------------------------------------
        # Update P-wave-related auxiliary stress field tau_p.
        #
        # tau_p uses the volumetric strain rate:
        #
        #   ∂vx/∂x + ∂vz/∂z
        #
        # This term is mainly associated with compressional wave motion.
        # ------------------------------------------------------------
        tau_p[iz, ix] += dt * (
            (lamb[iz, ix] + 2 * muon[iz, ix])
            * (xx[iz, ix] + zz[iz, ix])
        )

        # ------------------------------------------------------------
        # Update S-wave-related auxiliary stress field tau_s.
        #
        # tau_s uses a curl-related strain-rate term:
        #
        #   ∂vx/∂z - ∂vz/∂x
        #
        # This term is mainly associated with shear wave motion.
        # ------------------------------------------------------------
        tau_s[iz, ix] += dt * (
            muon[iz, ix] * (-zx[iz, ix] + xz[iz, ix])
        )