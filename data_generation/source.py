import numpy as np

def source(pfac, f, nt, dt):
    """
    Source definition function to generate a Ricker wavelet source.
    
    Parameters:
      pfac (float): Amplitude scaling factor for the source.
      f (float): Dominant frequency of the wavelet (in Hz).
      nt (int): Number of time steps.
      dt (float): Time step duration (in seconds).
    
    Returns:
      sour (numpy.ndarray): Time series of the source wavelet.
    """
    t0 = 0.1  # Time delay to center the wavelet
    t = np.arange(nt) * dt  # Create an array of time values for each time step

    B = np.pi * np.pi * f * f  # Precompute the constant B for the Ricker wavelet formula

    # Define the Ricker wavelet using its analytical expression:
    # The wavelet is given by: (0.5 - B*(t-t0)^2) * exp(-B*(t-t0)^2)
    SOrig = (0.5 - B * (t - t0) ** 2) * np.exp(-B * (t - t0) ** 2)
    sour = SOrig * pfac  # Scale the wavelet by the source amplitude factor

    return sour
