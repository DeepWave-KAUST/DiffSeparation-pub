import numpy as np
from numba import jit, prange
import math


# ================================================================
# Function: cpml_coefficients
# ================================================================
# Purpose:
#   Compute the CPML absorbing-boundary coefficients for 2D elastic
#   wave-equation modeling.
#
#   CPML stands for Convolutional Perfectly Matched Layer. It is used
#   to absorb outgoing waves near the computational boundaries and
#   reduce artificial boundary reflections.
#
#   This function computes the damping profiles and memory-variable
#   coefficients in both x and z directions:
#
#       ax, bx, kx : CPML coefficients in the x direction
#       az, bz, kz : CPML coefficients in the z direction
#
#   The resulting coefficients are later used in updatev.py and
#   updatep.py to correct spatial derivatives inside the absorbing
#   boundary region.
#
# Inputs:
#   ax, az : numpy.ndarray
#       CPML memory-variable scaling coefficients.
#       ax is used for x-direction derivatives.
#       az is used for z-direction derivatives.
#
#   bx, bz : numpy.ndarray
#       CPML exponential damping coefficients.
#       bx is used for x-direction derivatives.
#       bz is used for z-direction derivatives.
#
#   kx, kz : numpy.ndarray
#       CPML stretching coefficients in x and z directions.
#       In the current implementation, they are initialized as ones.
#
#   ddx, ddz : numpy.ndarray
#       Damping profiles in x and z directions.
#
#   alpx, alpz : numpy.ndarray
#       Alpha profiles in x and z directions. These terms help improve
#       absorption, especially for low-frequency components and grazing
#       incidence waves.
#
#   nx, nz : int
#       Number of physical grid points in x and z directions, excluding
#       CPML boundaries.
#
#   npd : int
#       Number of CPML grid points on each side of the model.
#
#   vp_max : float
#       Maximum P-wave velocity of the model. It is used to estimate the
#       maximum damping strength.
#
#   dx : float
#       Grid spacing. In this implementation, the same dx is used to
#       estimate the damping strength for both x and z directions.
#       This is acceptable when dx == dz.
#
#   favg : float
#       Dominant frequency of the source wavelet. It controls the maximum
#       alpha value.
#
#   dt : float
#       Time sampling interval.
#
# Outputs:
#   This function updates ax, az, bx, bz, ddx, ddz, alpx, and alpz in place.
#   It does not explicitly return values.
# ================================================================
@jit(nopython=True, parallel=True)
def cpml_coefficients(ax, az, bx, bz, kx, kz,
                      ddx, ddz, alpx, alpz,
                      nx, nz, npd, vp_max, dx, favg, dt):

    # Total number of grid points after padding the physical model
    # with CPML layers on all sides.
    #
    # pnx: total grid points in x direction
    # pnz: total grid points in z direction
    pnx = nx + 2 * npd
    pnz = nz + 2 * npd

    # ------------------------------------------------------------
    # CPML global parameters.
    # ------------------------------------------------------------

    # Target theoretical reflection coefficient at the outer boundary.
    # A smaller value means stronger absorption.
    R = 1e-11

    # Polynomial order of the damping profile.
    # pml_m = 2 means the damping increases quadratically toward
    # the outer boundary.
    pml_m = 2

    # Maximum damping coefficient.
    #
    # The damping strength depends on the maximum velocity, the CPML
    # thickness, grid spacing, and target reflection coefficient.
    dd_max = -((pml_m + 1) * vp_max * math.log(R) / (2 * npd * dx))

    # Maximum kappa values.
    # In the current implementation, kx and kz are initialized as ones,
    # so no additional kappa stretching is applied.
    kx_max = 1
    kz_max = 1

    # Maximum alpha values in x and z directions.
    # Alpha is usually used to improve absorption for low-frequency
    # and grazing-incidence waves.
    alpx_max = math.pi * favg
    alpz_max = math.pi * favg

    # ------------------------------------------------------------
    # Left CPML boundary.
    #
    # ix ranges from 0 to npd - 1.
    # The damping is strongest at the outermost boundary ix = 0
    # and gradually decreases toward the physical domain.
    # ------------------------------------------------------------
    for ix in prange(npd):
        for iz in range(pnz):

            # Polynomial damping profile in the x direction.
            ddx[iz, ix] = dd_max * (((npd - ix) / npd) ** pml_m)

            # Alpha profile in the x direction.
            alpx[iz, ix] = alpx_max * (1 - ((npd - ix) / npd))
            alpx[iz, ix] = max(alpx[iz, ix], 0.0)

            # Exponential damping coefficient.
            bx[iz, ix] = math.exp(
                -(ddx[iz, ix] / kx[iz, ix] + alpx[iz, ix]) * dt
            )

            # Memory-variable coefficient.
            if ddx[iz, ix] > 1e-6:
                ax[iz, ix] = (
                    ddx[iz, ix] * (bx[iz, ix] - 1)
                    / (
                        kx[iz, ix]
                        * (ddx[iz, ix] + kx[iz, ix] * alpx[iz, ix])
                    )
                )

    # ------------------------------------------------------------
    # Right CPML boundary.
    #
    # ix ranges from npd + nx to 2*npd + nx - 1.
    # The damping increases from the physical-domain interface toward
    # the outermost right boundary.
    # ------------------------------------------------------------
    for ix in prange(npd + nx, 2 * npd + nx):
        for iz in range(pnz):

            # Polynomial damping profile in the x direction.
            ddx[iz, ix] = dd_max * (
                ((ix - nx - npd + 1) / npd) ** pml_m
            )

            # Alpha profile in the x direction.
            alpx[iz, ix] = alpx_max * (
                1 - ((ix - nx - npd + 1) / npd)
            )
            alpx[iz, ix] = max(alpx[iz, ix], 0.0)

            # Exponential damping coefficient.
            bx[iz, ix] = math.exp(
                -(ddx[iz, ix] / kx[iz, ix] + alpx[iz, ix]) * dt
            )

            # Memory-variable coefficient.
            if ddx[iz, ix] > 1e-6:
                ax[iz, ix] = (
                    ddx[iz, ix] * (bx[iz, ix] - 1)
                    / (
                        kx[iz, ix]
                        * (ddx[iz, ix] + kx[iz, ix] * alpx[iz, ix])
                    )
                )

    # ------------------------------------------------------------
    # Top CPML boundary.
    #
    # iz ranges from 0 to npd - 1.
    # The damping is strongest at the top outer boundary and gradually
    # decreases toward the physical domain.
    # ------------------------------------------------------------
    for ix in prange(pnx):
        for iz in range(npd):

            # Polynomial damping profile in the z direction.
            ddz[iz, ix] = dd_max * (((npd - iz) / npd) ** pml_m)

            # Alpha profile in the z direction.
            alpz[iz, ix] = alpz_max * (1 - ((npd - iz) / npd))
            alpz[iz, ix] = max(alpz[iz, ix], 0.0)

            # Exponential damping coefficient.
            bz[iz, ix] = math.exp(
                -(ddz[iz, ix] / kz[iz, ix] + alpz[iz, ix]) * dt
            )

            # Memory-variable coefficient.
            if ddz[iz, ix] > 1e-6:
                az[iz, ix] = (
                    ddz[iz, ix] * (bz[iz, ix] - 1)
                    / (
                        kz[iz, ix]
                        * (ddz[iz, ix] + kz[iz, ix] * alpz[iz, ix])
                    )
                )

    # ------------------------------------------------------------
    # Bottom CPML boundary.
    #
    # iz ranges from npd + nz to 2*npd + nz - 1.
    # The damping increases from the physical-domain interface toward
    # the bottom outer boundary.
    # ------------------------------------------------------------
    for ix in prange(pnx):
        for iz in range(npd + nz, 2 * npd + nz):

            # Polynomial damping profile in the z direction.
            ddz[iz, ix] = dd_max * (
                ((iz - nz - npd + 1) / npd) ** pml_m
            )

            # Alpha profile in the z direction.
            alpz[iz, ix] = alpz_max * (
                1 - ((iz - nz - npd + 1) / npd)
            )
            alpz[iz, ix] = max(alpz[iz, ix], 0.0)

            # Exponential damping coefficient.
            bz[iz, ix] = math.exp(
                -(ddz[iz, ix] / kz[iz, ix] + alpz[iz, ix]) * dt
            )

            # Memory-variable coefficient.
            if ddz[iz, ix] > 1e-6:
                az[iz, ix] = (
                    ddz[iz, ix] * (bz[iz, ix] - 1)
                    / (
                        kz[iz, ix]
                        * (ddz[iz, ix] + kz[iz, ix] * alpz[iz, ix])
                    )
                )


# ================================================================
# Function: initialize_cpml
# ================================================================
# Purpose:
#   Initialize all CPML coefficient arrays on the CPU and then call
#   cpml_coefficients to fill the boundary coefficients.
#
#   The returned arrays are later transferred to the GPU in the main
#   modeling script and used by updatev.py and updatep.py.
#
# Inputs:
#   nx, nz : int
#       Number of physical grid points in x and z directions.
#
#   npd : int
#       Number of CPML grid points on each boundary side.
#
#   vp_max : float
#       Maximum P-wave velocity in the model.
#
#   dx : float
#       Spatial grid interval. This implementation assumes dx == dz
#       for the damping-strength calculation.
#
#   favg : float
#       Dominant frequency of the source wavelet.
#
#   dt : float
#       Time sampling interval.
#
# Outputs:
#   ax, az, bx, bz, kx, kz : numpy.ndarray
#       CPML coefficients and scaling arrays.
#       These arrays have shape:
#
#           (nz + 2*npd, nx + 2*npd)
# ================================================================
def initialize_cpml(nx, nz, npd, vp_max, dx, favg, dt):

    # Total model size after adding CPML layers.
    pnx = nx + 2 * npd
    pnz = nz + 2 * npd

    # ------------------------------------------------------------
    # Initialize CPML coefficient arrays in host memory.
    #
    # ax, az:
    #   Memory-variable coefficients.
    #
    # bx, bz:
    #   Exponential damping coefficients.
    #
    # ddx, ddz:
    #   Damping profiles. They are intermediate arrays used only
    #   during coefficient calculation.
    #
    # alpx, alpz:
    #   Alpha profiles. They are also intermediate arrays.
    #
    # kx, kz:
    #   Kappa/stretching profiles. Here they are initialized as ones,
    #   which means no additional kappa stretching is applied.
    # ------------------------------------------------------------
    ax = np.zeros((pnz, pnx), dtype=np.float32)
    az = np.zeros((pnz, pnx), dtype=np.float32)
    bx = np.zeros((pnz, pnx), dtype=np.float32)
    bz = np.zeros((pnz, pnx), dtype=np.float32)

    ddx = np.zeros((pnz, pnx), dtype=np.float32)
    ddz = np.zeros((pnz, pnx), dtype=np.float32)

    alpx = np.zeros((pnz, pnx), dtype=np.float32)
    alpz = np.zeros((pnz, pnx), dtype=np.float32)

    kx = np.ones((pnz, pnx), dtype=np.float32)
    kz = np.ones((pnz, pnx), dtype=np.float32)

    # Compute CPML coefficients in the boundary layers.
    # The arrays are updated in place by cpml_coefficients.
    cpml_coefficients(
        ax, az, bx, bz,
        kx, kz,
        ddx, ddz,
        alpx, alpz,
        nx, nz, npd,
        vp_max, dx, favg, dt
    )

    # Return the CPML coefficients that are needed by the CUDA kernels.
    return ax, az, bx, bz, kx, kz


# ================================================================
# Example usage
# ================================================================
# This block is only executed when running this file directly.
# It is not executed when this file is imported by the main modeling code.
# ================================================================
if __name__ == "__main__":

    # Physical model size.
    nx, nz = 100, 100

    # Number of CPML layers.
    npd = 10

    # Maximum P-wave velocity.
    vp_max = 3000

    # Grid spacing.
    dx = 10

    # Dominant source frequency.
    favg = 20

    # Time sampling interval.
    dt = 0.001

    # Initialize CPML coefficients.
    ax, az, bx, bz, kx, kz = initialize_cpml(
        nx, nz, npd, vp_max, dx, favg, dt
    )

    print("CPML coefficients computed successfully.")