import numpy as np
from numba import cuda


# ================================================================
# CUDA kernel: updatev1_cuda
# ================================================================
# Purpose:
#   This CUDA kernel computes the spatial derivatives required for
#   updating the elastic velocity wavefield.
#
#   Specifically, it computes:
#
#       xxx = ∂pxx/∂x
#       xzz = ∂pxz/∂z
#       xzx = ∂pxz/∂x
#       zzz = ∂pzz/∂z
#
#   These terms are later used to update the horizontal and vertical
#   particle velocity components:
#
#       vx += dt * (∂pxx/∂x + ∂pxz/∂z) / rho
#       vz += dt * (∂pxz/∂x + ∂pzz/∂z) / rho
#
#   In addition, this kernel also computes the spatial derivatives of
#   tau_p and tau_s, which are used to update the separated P-wave and
#   S-wave velocity components.
#
# Inputs:
#   shape : tuple
#       Shape of the padded computational domain, including CPML layers.
#
#   dx, dz : float
#       Spatial grid intervals in the x and z directions.
#
#   vz, vx : device arrays
#       Vertical and horizontal particle velocity components.
#       These variables are passed to keep a consistent interface, but
#       they are not directly updated in this first kernel.
#
#   pxx, pzz, pxz : device arrays
#       Stress components of the elastic wavefield.
#
#   rho : device array
#       Density model. It is not directly used in this kernel, but is
#       used in updatev2_cuda.
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
#       CPML parameters. They are passed here for interface consistency
#       but are mainly used in updatev2_cuda.
#
# Outputs:
#   xxx, xzz, xzx, zzz : device arrays
#       Spatial derivatives of stress components used for velocity update.
#
#   tau_px, tau_pz : device arrays
#       Spatial derivatives of tau_p, associated with the P-wave component.
#
#   tau_sx, tau_sz : device arrays
#       Spatial derivatives of tau_s, associated with the S-wave component.
# ================================================================
@cuda.jit
def updatev1_cuda(shape, dx, dz, vz, vx, pxx, pzz, pxz, rho, dt, coef, kk,
                 ax, az, bx, bz, kxc, kzc,
                 pml_pxxx, pml_pxzz, pml_pxzx, pml_pzzz, tau_p, s_px, s_pz,
                 pml_tau_px, pml_tau_pz, tau_s, s_sx, s_sz,
                 pml_tau_sx, pml_tau_sz,
                 xxx, xzz, xzx, zzz, tau_px, tau_pz, tau_sx, tau_sz):

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
        # Compute stress derivatives:
        #
        #   xxx = ∂pxx/∂x
        #   xzz = ∂pxz/∂z
        #
        # These two terms will be used to update the horizontal velocity vx:
        #
        #   vx += dt * (xxx + xzz) / rho
        #
        # The finite-difference stencil follows a staggered-grid form.
        # ------------------------------------------------------------
        diff1, diff2 = 0.0, 0.0

        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (
                pxx[iz, ix + ii - 1] - pxx[iz, ix - ii]
            )

            diff2 += coef[ii - 1] * (
                pxz[iz + ii - 1, ix] - pxz[iz - ii, ix]
            )

        xxx[iz, ix] = diff1 / dx
        xzz[iz, ix] = diff2 / dz

        # ------------------------------------------------------------
        # Compute stress derivatives:
        #
        #   xzx = ∂pxz/∂x
        #   zzz = ∂pzz/∂z
        #
        # These two terms will be used to update the vertical velocity vz:
        #
        #   vz += dt * (xzx + zzz) / rho
        # ------------------------------------------------------------
        diff1 = 0.0
        diff2 = 0.0

        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (
                pxz[iz, ix + ii] - pxz[iz, ix - ii + 1]
            )

            diff2 += coef[ii - 1] * (
                pzz[iz + ii, ix] - pzz[iz - ii + 1, ix]
            )

        xzx[iz, ix] = diff1 / dx
        zzz[iz, ix] = diff2 / dz

        # ------------------------------------------------------------
        # Compute derivatives of tau_p:
        #
        #   tau_px = ∂tau_p/∂x
        #   tau_pz = ∂tau_p/∂z
        #
        # tau_p is related to the P-wave component. Its spatial derivatives
        # are later used to update the separated P-wave velocity components:
        #
        #   s_px += dt * tau_px / rho
        #   s_pz += dt * tau_pz / rho
        # ------------------------------------------------------------
        diff1, diff2 = 0.0, 0.0

        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (
                tau_p[iz, ix + ii - 1] - tau_p[iz, ix - ii]
            )

            diff2 += coef[ii - 1] * (
                tau_p[iz + ii, ix] - tau_p[iz - ii + 1, ix]
            )

        tau_px[iz, ix] = diff1 / dx
        tau_pz[iz, ix] = diff2 / dz

        # ------------------------------------------------------------
        # Compute derivatives of tau_s:
        #
        #   tau_sx = ∂tau_s/∂z
        #   tau_sz = -∂tau_s/∂x
        #
        # tau_s is related to the S-wave component. Its spatial derivatives
        # are later used to update the separated S-wave velocity components:
        #
        #   s_sx += dt * tau_sx / rho
        #   s_sz += dt * tau_sz / rho
        #
        # The negative sign in tau_sz follows the curl-related formulation
        # used in the elastic wavefield separation.
        # ------------------------------------------------------------
        diff1 = 0.0
        diff2 = 0.0

        for ii in range(1, kk + 1):
            diff1 += coef[ii - 1] * (
                tau_s[iz + ii - 1, ix] - tau_s[iz - ii, ix]
            )

            diff2 += coef[ii - 1] * (
                tau_s[iz, ix + ii] - tau_s[iz, ix - ii + 1]
            )

        tau_sx[iz, ix] = diff1 / dz
        tau_sz[iz, ix] = -diff2 / dx


# ================================================================
# CUDA kernel: updatev2_cuda
# ================================================================
# Purpose:
#   This CUDA kernel applies CPML corrections to the spatial derivative
#   terms computed in updatev1_cuda, and then updates:
#
#       1. The full elastic velocity wavefield:
#              vx, vz
#
#       2. The separated P-wave velocity components:
#              s_px, s_pz
#
#       3. The separated S-wave velocity components:
#              s_sx, s_sz
#
#   The CPML memory variables are updated before the velocity update,
#   so that wavefield absorption near the computational boundary is
#   properly included.
#
# Inputs:
#   xxx, xzz, xzx, zzz :
#       Stress derivative terms computed by updatev1_cuda.
#
#   tau_px, tau_pz :
#       Spatial derivatives of tau_p for updating P-wave components.
#
#   tau_sx, tau_sz :
#       Spatial derivatives of tau_s for updating S-wave components.
#
#   ax, az, bx, bz, kxc, kzc :
#       CPML coefficients used to modify the spatial derivatives inside
#       the absorbing boundary region.
#
#   pml_* :
#       CPML memory variables for stress derivatives and tau derivatives.
#
# Outputs:
#   vx, vz :
#       Updated horizontal and vertical velocity components.
#
#   s_px, s_pz :
#       Updated separated P-wave velocity components.
#
#   s_sx, s_sz :
#       Updated separated S-wave velocity components.
# ================================================================
@cuda.jit
def updatev2_cuda(shape, dx, dz, vz, vx, pxx, pzz, pxz, rho, dt, coef, kk,
                 ax, az, bx, bz, kxc, kzc,
                 pml_pxxx, pml_pxzz, pml_pxzx, pml_pzzz, tau_p, s_px, s_pz,
                 pml_tau_px, pml_tau_pz, tau_s, s_sx, s_sz,
                 pml_tau_sx, pml_tau_sz,
                 xxx, xzz, xzx, zzz, tau_px, tau_pz, tau_sx, tau_sz):

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
        # Apply CPML correction to xxx = ∂pxx/∂x.
        #
        # pml_pxxx is the CPML memory variable for this derivative.
        # kxc is the CPML scaling factor in the x direction.
        # ------------------------------------------------------------
        pml_pxxx[iz, ix] = (
            bx[iz, ix] * pml_pxxx[iz, ix]
            + ax[iz, ix] * xxx[iz, ix]
        )
        xxx[iz, ix] = xxx[iz, ix] / kxc[iz, ix] + pml_pxxx[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to xzz = ∂pxz/∂z.
        #
        # pml_pxzz is the CPML memory variable for this derivative.
        # kzc is the CPML scaling factor in the z direction.
        # ------------------------------------------------------------
        pml_pxzz[iz, ix] = (
            bz[iz, ix] * pml_pxzz[iz, ix]
            + az[iz, ix] * xzz[iz, ix]
        )
        xzz[iz, ix] = xzz[iz, ix] / kzc[iz, ix] + pml_pxzz[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to xzx = ∂pxz/∂x.
        # ------------------------------------------------------------
        pml_pxzx[iz, ix] = (
            bx[iz, ix] * pml_pxzx[iz, ix]
            + ax[iz, ix] * xzx[iz, ix]
        )
        xzx[iz, ix] = xzx[iz, ix] / kxc[iz, ix] + pml_pxzx[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to zzz = ∂pzz/∂z.
        # ------------------------------------------------------------
        pml_pzzz[iz, ix] = (
            bz[iz, ix] * pml_pzzz[iz, ix]
            + az[iz, ix] * zzz[iz, ix]
        )
        zzz[iz, ix] = zzz[iz, ix] / kzc[iz, ix] + pml_pzzz[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to tau_px = ∂tau_p/∂x.
        #
        # This corrected term is used to update the x-component of the
        # separated P-wave velocity field.
        # ------------------------------------------------------------
        pml_tau_px[iz, ix] = (
            bx[iz, ix] * pml_tau_px[iz, ix]
            + ax[iz, ix] * tau_px[iz, ix]
        )
        tau_px[iz, ix] = tau_px[iz, ix] / kxc[iz, ix] + pml_tau_px[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to tau_pz = ∂tau_p/∂z.
        #
        # This corrected term is used to update the z-component of the
        # separated P-wave velocity field.
        # ------------------------------------------------------------
        pml_tau_pz[iz, ix] = (
            bz[iz, ix] * pml_tau_pz[iz, ix]
            + az[iz, ix] * tau_pz[iz, ix]
        )
        tau_pz[iz, ix] = tau_pz[iz, ix] / kzc[iz, ix] + pml_tau_pz[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to tau_sx.
        #
        # This corrected term is used to update the x-component of the
        # separated S-wave velocity field.
        # ------------------------------------------------------------
        pml_tau_sx[iz, ix] = (
            bx[iz, ix] * pml_tau_sx[iz, ix]
            + ax[iz, ix] * tau_sx[iz, ix]
        )
        tau_sx[iz, ix] = tau_sx[iz, ix] / kxc[iz, ix] + pml_tau_sx[iz, ix]

        # ------------------------------------------------------------
        # Apply CPML correction to tau_sz.
        #
        # This corrected term is used to update the z-component of the
        # separated S-wave velocity field.
        # ------------------------------------------------------------
        pml_tau_sz[iz, ix] = (
            bz[iz, ix] * pml_tau_sz[iz, ix]
            + az[iz, ix] * tau_sz[iz, ix]
        )
        tau_sz[iz, ix] = tau_sz[iz, ix] / kzc[iz, ix] + pml_tau_sz[iz, ix]

        # ------------------------------------------------------------
        # Update the full elastic velocity wavefield.
        #
        # Horizontal velocity:
        #   vx = vx + dt * (∂pxx/∂x + ∂pxz/∂z) / rho
        #
        # Vertical velocity:
        #   vz = vz + dt * (∂pxz/∂x + ∂pzz/∂z) / rho
        # ------------------------------------------------------------
        vx[iz, ix] += dt * (xxx[iz, ix] + xzz[iz, ix]) / rho[iz, ix]
        vz[iz, ix] += dt * (xzx[iz, ix] + zzz[iz, ix]) / rho[iz, ix]

        # ------------------------------------------------------------
        # Update separated P-wave velocity components.
        #
        # s_px and s_pz are obtained by integrating the corrected
        # derivatives of tau_p over time.
        # ------------------------------------------------------------
        s_px[iz, ix] += dt * tau_px[iz, ix] / rho[iz, ix]
        s_pz[iz, ix] += dt * tau_pz[iz, ix] / rho[iz, ix]

        # ------------------------------------------------------------
        # Update separated S-wave velocity components.
        #
        # s_sx and s_sz are obtained by integrating the corrected
        # derivatives of tau_s over time.
        # ------------------------------------------------------------
        s_sx[iz, ix] += dt * tau_sx[iz, ix] / rho[iz, ix]
        s_sz[iz, ix] += dt * tau_sz[iz, ix] / rho[iz, ix]