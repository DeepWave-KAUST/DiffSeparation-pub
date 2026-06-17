import numpy as np


# ================================================================
# Function: source
# ================================================================
# Purpose:
#   Generate a Ricker-like source time function for seismic wave modeling.
#
#   This source function is commonly used as the time-dependent source
#   term in finite-difference wave-equation simulations. In the main
#   modeling code, the generated source wavelet is injected into the
#   vertical velocity component vz at the source location.
#
# Mathematical form:
#   Let:
#
#       B = pi^2 * f^2
#
#   The source wavelet is defined as:
#
#       S(t) = [0.5 - B * (t - t0)^2] * exp[-B * (t - t0)^2]
#
#   Then the final source is scaled by pfac:
#
#       sour(t) = pfac * S(t)
#
# Inputs:
#   pfac : float
#       Source amplitude scaling factor. A larger pfac produces a stronger
#       injected source.
#
#   f : float
#       Dominant frequency of the source wavelet, in Hz.
#
#   nt : int
#       Number of time samples.
#
#   dt : float
#       Time sampling interval, in seconds.
#
# Output:
#   sour : numpy.ndarray
#       Source time series with length nt.
# ================================================================
def source(pfac, f, nt, dt):
    """
    Generate a Ricker-like seismic source wavelet.

    Parameters
    ----------
    pfac : float
        Source amplitude scaling factor.

    f : float
        Dominant frequency of the source wavelet, in Hz.

    nt : int
        Number of time samples.

    dt : float
        Time sampling interval, in seconds.

    Returns
    -------
    sour : numpy.ndarray
        Source time series with shape (nt,).
    """

    # Time delay of the source wavelet.
    # This shifts the peak of the source away from t = 0 to avoid
    # starting the simulation with a nonzero or abruptly varying source.
    t0 = 0.1

    # Generate the discrete time axis:
    #
    #   t = [0, dt, 2dt, ..., (nt-1)dt]
    #
    # The length of t is nt.
    t = np.arange(nt) * dt

    # Frequency-dependent coefficient used in the Ricker-like wavelet.
    #
    #   B = pi^2 * f^2
    #
    # where f is the dominant frequency.
    B = np.pi * np.pi * f * f

    # Generate the unscaled source wavelet.
    #
    # Compared with the standard Ricker wavelet form:
    #
    #   (1 - 2B(t - t0)^2) * exp[-B(t - t0)^2]
    #
    # this implementation uses:
    #
    #   (0.5 - B(t - t0)^2) * exp[-B(t - t0)^2]
    #
    # Therefore, it has a similar Ricker-like shape but with a different
    # amplitude scaling.
    SOrig = (0.5 - B * (t - t0) ** 2) * np.exp(-B * (t - t0) ** 2)

    # Apply the source amplitude scaling factor.
    sour = SOrig * pfac

    # Return the final source time series.
    return sour