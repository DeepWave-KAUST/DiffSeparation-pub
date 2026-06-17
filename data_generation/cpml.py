import numpy as np
from numba import jit, prange
import math

# Use Numba's JIT compilation for faster execution with parallel loops
@jit(nopython=True, parallel=True)
def cpml_coefficients(ax, az, bx, bz, kx, kz, ddx, ddz, alpx, alpz,
                      nx, nz, npd, vp_max, dx, favg, dt):
    """
    Compute CPML (Convolutional Perfectly Matched Layer) coefficients for the simulation.
    
    Parameters:
      ax, az: Arrays to store the damping coefficients for the x and z directions.
      bx, bz: Arrays to store the exponential damping factors for the x and z directions.
      kx, kz: Arrays representing scaling factors (typically set to 1) for the x and z directions.
      ddx, ddz: Arrays to store the damping profiles along the x and z directions.
      alpx, alpz: Arrays to store the alpha parameters (frequency shifts) for x and z.
      nx, nz: Number of grid points in the x and z directions (without CPML layers).
      npd: Number of grid points used for the CPML (damping) layers.
      vp_max: Maximum P-wave velocity in the medium.
      dx: Spatial grid spacing.
      favg: Central frequency of the source wavelet.
      dt: Time step size.
    """
    # Compute total grid dimensions including CPML layers
    pnx = nx + 2 * npd
    pnz = nz + 2 * npd

    # Define CPML parameters
    R = 1e-11             # Target reflection coefficient
    pml_m = 2             # CPML power order
    # Maximum damping value computed from desired reflection and grid parameters
    dd_max = -((pml_m + 1) * vp_max * math.log(R) / (2 * npd * dx))
    kx_max = 1            # Maximum value for kx (not modified in this function)
    kz_max = 1            # Maximum value for kz (not modified in this function)
    alpx_max = math.pi * favg  # Maximum alpha value in x direction
    alpz_max = math.pi * favg  # Maximum alpha value in z direction

    # --------------------------
    # Left Side CPML (for x indices 0 to npd-1)
    # --------------------------
    for ix in prange(npd):           # Loop over CPML region on the left
        for iz in range(pnz):         # Loop over the full vertical dimension
            # Compute the damping profile for x-direction
            ddx[iz, ix] = dd_max * (((npd - ix) / npd) ** pml_m)
            # Compute the alpha parameter for x-direction, reducing near the interior
            alpx[iz, ix] = alpx_max * (1 - ((npd - ix) / npd))
            # Ensure that alpha is non-negative
            alpx[iz, ix] = max(alpx[iz, ix], 0.0)
            # Compute the exponential damping factor for x-direction
            bx[iz, ix] = math.exp(-(ddx[iz, ix] / kx[iz, ix] + alpx[iz, ix]) * dt)
            # Compute the damping coefficient if the damping value is significant
            if ddx[iz, ix] > 1e-6:
                ax[iz, ix] = (ddx[iz, ix] * (bx[iz, ix] - 1)) / (kx[iz, ix] * (ddx[iz, ix] + kx[iz, ix] * alpx[iz, ix]))

    # --------------------------
    # Right Side CPML (for x indices nx+npd to nx+2*npd-1)
    # --------------------------
    for ix in prange(npd + nx, 2 * npd + nx):  # Loop over CPML region on the right
        for iz in range(pnz):
            # Compute damping profile for x-direction on the right side
            ddx[iz, ix] = dd_max * (((ix - nx - npd + 1) / npd) ** pml_m)
            # Compute the alpha parameter for the right side in x-direction
            alpx[iz, ix] = alpx_max * (1 - ((ix - nx - npd + 1) / npd))
            alpx[iz, ix] = max(alpx[iz, ix], 0.0)
            # Compute the exponential damping factor for the right side in x-direction
            bx[iz, ix] = math.exp(-(ddx[iz, ix] / kx[iz, ix] + alpx[iz, ix]) * dt)
            if ddx[iz, ix] > 1e-6:
                ax[iz, ix] = (ddx[iz, ix] * (bx[iz, ix] - 1)) / (kx[iz, ix] * (ddx[iz, ix] + kx[iz, ix] * alpx[iz, ix]))

    # --------------------------
    # Top Side CPML (for z indices 0 to npd-1)
    # --------------------------
    for ix in prange(pnx):           # Loop over the entire x direction
        for iz in range(npd):         # Loop over CPML region on the top
            # Compute the damping profile for z-direction (top side)
            ddz[iz, ix] = dd_max * (((npd - iz) / npd) ** pml_m)
            # Compute the alpha parameter for z-direction (top side)
            alpz[iz, ix] = alpz_max * (1 - ((npd - iz) / npd))
            alpz[iz, ix] = max(alpz[iz, ix], 0.0)
            # Compute the exponential damping factor for z-direction (top side)
            bz[iz, ix] = math.exp(-(ddz[iz, ix] / kz[iz, ix] + alpz[iz, ix]) * dt)
            if ddz[iz, ix] > 1e-6:
                az[iz, ix] = (ddz[iz, ix] * (bz[iz, ix] - 1)) / (kz[iz, ix] * (ddz[iz, ix] + kz[iz, ix] * alpz[iz, ix]))

    # --------------------------
    # Bottom Side CPML (for z indices nz+npd to nz+2*npd-1)
    # --------------------------
    for ix in prange(pnx):
        for iz in range(npd + nz, 2 * npd + nz):  # Loop over CPML region on the bottom
            # Compute the damping profile for z-direction (bottom side)
            ddz[iz, ix] = dd_max * (((iz - nz - npd + 1) / npd) ** pml_m)
            # Compute the alpha parameter for z-direction (bottom side)
            alpz[iz, ix] = alpz_max * (1 - ((iz - nz - npd + 1) / npd))
            alpz[iz, ix] = max(alpz[iz, ix], 0.0)
            # Compute the exponential damping factor for z-direction (bottom side)
            bz[iz, ix] = math.exp(-(ddz[iz, ix] / kz[iz, ix] + alpz[iz, ix]) * dt)
            if ddz[iz, ix] > 1e-6:
                az[iz, ix] = (ddz[iz, ix] * (bz[iz, ix] - 1)) / (kz[iz, ix] * (ddz[iz, ix] + kz[iz, ix] * alpz[iz, ix]))

def initialize_cpml(nx, nz, npd, vp_max, dx, favg, dt):
    """
    Initialize the CPML arrays and compute the CPML coefficients.
    
    Parameters:
      nx, nz: Number of grid points in the x and z directions (without CPML layers).
      npd: Number of CPML grid points.
      vp_max: Maximum P-wave velocity.
      dx: Spatial grid spacing.
      favg: Central frequency of the source wavelet.
      dt: Time step size.
      
    Returns:
      ax, az: Damping coefficients in x and z directions.
      bx, bz: Exponential damping factors in x and z directions.
      kx, kz: Scaling factors (typically ones) in x and z directions.
    """
    # Compute total grid dimensions including CPML layers
    pnx = nx + 2 * npd
    pnz = nz + 2 * npd

    # Initialize CPML coefficient arrays in host memory
    ax = np.zeros((pnz, pnx), dtype=np.float32)
    az = np.zeros((pnz, pnx), dtype=np.float32)
    bx = np.zeros((pnz, pnx), dtype=np.float32)
    bz = np.zeros((pnz, pnx), dtype=np.float32)
    ddx = np.zeros((pnz, pnx), dtype=np.float32)
    ddz = np.zeros((pnz, pnx), dtype=np.float32)
    alpx = np.zeros((pnz, pnx), dtype=np.float32)
    alpz = np.zeros((pnz, pnx), dtype=np.float32)
    # kx and kz are usually set to one throughout the domain
    kx = np.ones((pnz, pnx), dtype=np.float32)
    kz = np.ones((pnz, pnx), dtype=np.float32)

    # Compute the CPML coefficients using the accelerated function
    cpml_coefficients(ax, az, bx, bz, kx, kz, ddx, ddz, alpx, alpz,
                      nx, nz, npd, vp_max, dx, favg, dt)

    return ax, az, bx, bz, kx, kz

# Example usage of the CPML initialization functions
if __name__ == "__main__":
    # Define simulation parameters
    nx, nz = 100, 100     # Number of grid points in x and z (without CPML)
    npd = 10              # Number of CPML grid points on each side
    vp_max = 3000         # Maximum P-wave velocity
    dx = 10               # Grid spacing in x direction
    favg = 20             # Central frequency of the source wavelet
    dt = 0.001            # Time step size

    # Initialize CPML coefficients
    ax, az, bx, bz, kx, kz = initialize_cpml(nx, nz, npd, vp_max, dx, favg, dt)
    print("CPML coefficients computed successfully.")
