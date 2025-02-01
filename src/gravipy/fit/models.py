import numpy as np

#Integration tools
from ..tools.complex_quadratures import complex_quadrature_num

def spectral_visibility_integrator(l, opd, alpha, reference_l0 = 2.2 ):
    """Integrand for the spectral visibility

    Args:
        l (float): Wavelength in micrometers
        opd (float): Optical path difference in micrometers. In interferometric terms this would be dot(baseline, sky_vector)
        alpha (float): Spectral index
        l0 (float, optional): Reference wavelength in micrometers. Defaults to 2.2.
    """
    return (l/reference_l0)**(-1-alpha)*np.exp(-2*np.pi*1j*opd/l)

def spectral_visibility(opd, alpha, l0=2.2, dl=0.2, reference_l0=2.2):
    """Spectral Visibility

    Args:
        opd (float): Optical path difference in micrometers. In interferometric terms this would be dot(baseline, sky_vector)
        alpha (float): Spectral index
        l0 (float, optional): Reference wavelength in micrometers. Defaults to 2.2.
        dl (float, optional): Half filter width in micrometers. Defaults to 0.2.
    """

    if np.all(opd == 0.) and alpha != 0:
        return  -reference_l0**(1 + alpha)/alpha * ( (l0+dl)**(-alpha) - (l0-dl)**(-alpha) )
    else:
        return complex_quadrature_num(spectral_visibility_integrator, l0-dl, l0+dl, (opd, alpha, reference_l0))

def nsource_visibility( bij, sources, background, l0=2.2, dl=0.2, reference_l0=2.2):
    
    visibility = 0.0
    normalization = 0.0

    for src in sources:
        
        #Get position, flux and spectral index for each source
        x, y, flux, alpha = src

        #Calculate optical path difference from position on sky and visibility
        s = bij[0]*x + bij[1]*y
        
        #Add source visibility to nsource one
        visibility += flux*spectral_visibility(s, alpha, l0, dl, reference_l0)
        normalization += flux*spectral_visibility(0, alpha, l0, dl, reference_l0)

    #Add background to normalization
    flux, alpha = background
    normalization += flux*spectral_visibility(0, alpha, l0, dl,reference_l0)
    
    return visibility/normalization
