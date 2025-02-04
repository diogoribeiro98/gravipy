import numpy as np

#Integration tools
from ..tools.complex_quadratures import complex_quadrature_num

def spectral_visibility_integrator(l, opd, alpha, reference_l0=2.2 ):
    """Integrand for the spectral visibility

    Args:
        l (float): Wavelength in micrometers
        opd (float): Optical path difference in micrometers. In interferometric terms this would be dot(baseline, sky_vector)
        alpha (float): Spectral index
        reference_l0 (float, optional): Reference wavelength in micrometers. Defaults to 2.2.

    Returns:
        float: Spectral visibility
    """    
    return (l/reference_l0)**(-1-alpha)*np.exp(-2*np.pi*1j*opd/l)

def spectral_visibility(opd, alpha, l, dl, reference_l0=2.2):
    """Spectral Visibility

    Args:
        opd (float): Optical path difference in micrometers. In interferometric terms this would be dot(baseline, sky_vector)
        alpha (float): Spectral index
        l (float): Central wavelength in micrometers
        dl (float, optional): Half bandwidth in micrometers. Defaults to 0.2.
        reference_l0 (float, optional): Reference wavelength in micrometers. Defaults to 2.2.
    """

    if np.all(opd == 0.) and alpha != 0:
        return  -reference_l0**(1 + alpha)/alpha * ( (l+dl)**(-alpha) - (l-dl)**(-alpha) )
    else:
        return complex_quadrature_num(spectral_visibility_integrator, l-dl, l+dl, (opd, alpha, reference_l0))
