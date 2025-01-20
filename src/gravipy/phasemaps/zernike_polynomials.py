import numpy as np

def zernike_polynomial(n,m,rho,phi):
	
	#Normalization factor
	if m==0:
		rms = np.sqrt((n+1))
	else:
		rms = np.sqrt(2*(n+1))
	
	match (n,m):	
		case (0,0):
			return 1
		
		case (1,-1):
			return rms*rho*np.sin(phi)
		case (1,1):
			return rms*rho*np.cos(phi)
		
		case (2,-2):
			return rms*(rho**2)*np.sin(2*phi)
		case (2,0):
			return rms*(2.*rho**2 -1)
		case (2,2):
			return rms*(rho**2)*np.cos(2*phi)
		
		case (3,-3):
			return rms*(rho**3)*np.sin(3*phi)
		case (3,-1):
			return rms*(3*rho**3-2*rho)*np.sin(phi)
		case (3,1):
			return rms*(3*rho**3-2*rho)*np.cos(phi)
		case (3,3):
			return rms*(rho**3)*np.cos(3*phi)
		
		case (4,-4):
			return rms*(rho**4)*np.sin(4*phi)
		case (4,-2):
			return rms*(4*rho**4-3*rho**2)*np.sin(2*phi)
		case (4,0):
			return rms*(6*rho**4-6*rho**2+1)
		case (4,2):
			return rms*(4*rho**4-3*rho**2)*np.cos(2*phi)
		case (4,4):
			return rms*(rho**4)*np.cos(4*phi)
		
		case (5,-5):
			return rms*(rho**5)*np.sin(5*phi)
		case (5,-3):
			return rms*(5*rho**5 - 4*rho**3)*np.sin(3*phi)
		case (5,-1):
			return rms*(10*rho**5 - 12*rho**3 + 3*rho)*np.sin(phi)
		case (5,1):
			return rms*(10*rho**5 - 12*rho**3 + 3*rho)*np.cos(phi)
		case (5,3):
			return rms*(5*rho**5 - 4*rho**3)*np.cos(3*phi)
		case (5,5):
			return rms*(rho**5)*np.cos(5*phi)
		
		# ATTENTION! 
		# The (6,-2) and (6,+2) polinomials defined below do not follow the zernike convention (wrong sign in the 6x^2 term) 
		# The original code to fit the phasemaps with a zernike decomposition used this definition and so we need to keep it
		case(6,-6):
			return rms*(rho**6)*np.sin(6*phi)
		case(6,-4):
			return rms*(6*rho**6 - 5*rho**4)*np.sin(4*phi)
		case(6,-2):
			return rms*(15*rho**6 - 20*rho**4 - 6*rho**2)*np.sin(2*phi)
		case(6,0):
			return rms*(20*rho**6 - 30*rho**4 + 12*rho**2-1)
		case(6,2):
			return rms*(15*rho**6 - 20*rho**4 - 6*rho**2)*np.cos(2*phi)
		case(6,4):
			return rms*(6*rho**6 - 5*rho**4)*np.cos(4*phi)
		case(6,6):
			return rms*(rho**6)*np.cos(6*phi)
		
		case _:
			raise ValueError("n must be smaller than 6, and |m| must be smaler than n.")
	