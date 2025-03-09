import numpy as np

def elliptical_beam(
        x, y, 
    	amplitude=1., 
    	sigmax=1., sigmay=1.,
    	theta=0):
    
	cs, sn = np.cos(theta), np.sin(theta)

	a = (  (cs/sigmax)**2 +(sn/sigmay)**2  ) 
	b = -cs*sn*((1/sigmax)**2 - (1/sigmay)**2 )
	c = (  (sn/sigmax)**2 +(cs/sigmay)**2  ) 

	R = -0.5*(a*x**2 + 2*b*x*y + c*y**2)

	return amplitude*np.exp(R)

def elliptical_beam_abc(
        x, y, 
    	A=1., 
    	Ax=1., Axy=0., Ay=1.):
    
	R = -(Ax*x**2 + 2*Axy*x*y + Ay*y**2)
	
	return A*np.exp(R)

def estimate_npoints(window, Ax, Axy, Ay, pixels_per_beam=5):
    """Estimate npoints for an elliptical Gaussian beam, matching np.linspace(-window, window, 2*npoints+1).
    
    Args:
        window (float): Image window size in arcseconds (or milliarcseconds).
        Ax (float): Gaussian coefficient for x^2.
        Axy (float): Gaussian cross-term coefficient for xy.
        Ay (float): Gaussian coefficient for y^2.
        pixels_per_beam (int): Desired pixels per beam (default=5).
    
    Returns:
        int: Estimated npoints.
    """
    # Compute eigenvalues of the covariance matrix
    lambda_plus = (Ax + Ay) / 2 + np.sqrt((Ax - Ay)**2 / 4 + Axy**2)
    lambda_minus = (Ax + Ay) / 2 - np.sqrt((Ax - Ay)**2 / 4 + Axy**2)

    # Convert eigenvalues to standard deviations
    sigma1 = 1 / np.sqrt(lambda_plus)
    sigma2 = 1 / np.sqrt(lambda_minus)

    # Convert to FWHM
    fwhm1 = 2 * np.sqrt(np.log(2)) * sigma1
    fwhm2 = 2 * np.sqrt(np.log(2)) * sigma2

    # Take the average FWHM
    fwhm_avg = (fwhm1 + fwhm2) / 2

    # Compute npoints using your np.linspace setup
    npoints = int((2 * window / fwhm_avg) * pixels_per_beam)

    return max(npoints, 10)  # Ensure a reasonable minimum
