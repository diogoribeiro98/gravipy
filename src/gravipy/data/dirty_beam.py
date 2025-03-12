import numpy as np
from scipy.signal import fftconvolve

from ..physical_units import units 
from ..tools.interferometric_beam import elliptical_beam_abc, estimate_npoints

from .interferometric_data_class import InterferometricData

def get_dirty_beam(
		idata 			: InterferometricData,
		window 			: float = 75,
		gain 			: float = 1.0, 
		threshold 		: float = 1e-3, 
		max_iter 		: float = None,
		pixels_per_beam : int 	=  2
		):
	"""
		Returns the Dirty Beam, Dirty Image and Clean Map of interferometric data.
		The function implements the CLEAN Algorithm introduced by Högbom (1974) in its original form.
	
	Args:
		idata (InterferometricData): Interferometric data
		window (float, optional): Size of reconstructed image in miliarcseconds. Defaults to 75.
		gain (float, optional): CLEAN loop gain. Defaults to 1.0.
		threshold (float, optional): Threshold maximum value to stop algorithm. Defaults to 1e-3.
		max_iter (float, optional): Maximum iterations. Defaults to None in which case it reduces to the number of points divided by the gain
		pixels_per_beam (int, optional): Number of pixels per central beam. Defaults to 2.

	Returns:
		(bx, B.real) : Dirty Beam 
		(x, I.real)  : Dirty Image 
		(x,clean_map.real): Clean Map	
	"""

	#Fetch data
	uv_coordinates = np.transpose([idata.Bu,idata.Bv])

	#
	# Calculate central interferometric beam and appropriate image size
	#

	#Central lobe
	Npoints, Ax, Axy, Ay = 0, 0, 0, 0

	for idx,coords in enumerate(uv_coordinates):
		for jj, wave in enumerate(idata.wave):
		
			if idata.vis_flag[idx][jj]:
				continue

			#uv dimensionless coordinates
			u,v = (coords / (wave * units.micrometer) )

			#Central lobe
			Npoints   +=1
			Ax  += u**2
			Axy += u*v
			Ay  += v**2

	#Evaluate central lobe gaussian parameters
	scale_factor = (2*np.pi**2/Npoints)*units.mas_to_rad**2
	Ax  *=scale_factor
	Axy *=scale_factor
	Ay  *=scale_factor

	#Select appropriate number of pixels
	npoints = estimate_npoints(window,Ax,Axy,Ay,pixels_per_beam)

	#
	# Calculate Dirty image, Dirty beam and Central PSF 
	#

	#Dirty image
	x = np.linspace(-window, window, 2*npoints+1)
	X, Y = np.meshgrid(x, x)  # Centers
	I = np.zeros_like(X)
	
	#Dirty beam (twice as big as the dirty image)
	xb = np.linspace(-2*window, 2*window, 4*npoints+1)
	Xb, Yb = np.meshgrid(xb, xb)
	B = np.zeros_like(Xb)
	
	for idx,coords in enumerate(uv_coordinates):
		for jj, wave in enumerate(idata.wave):
		
			if idata.vis_flag[idx][jj]:
				continue

			#uv dimensionless coordinates
			u,v = (coords / (wave * units.micrometer)) * units.mas_to_rad 

			#Visibility
			amp   = idata.visamp[idx][jj]
			phase = np.deg2rad(idata.visphi[idx][jj])  
			
			#Dirty beam
			B += np.real( np.exp(2*np.pi*1j*( Xb*u + Yb*v )))

			#Dirty image
			V = amp*np.exp(1j*phase)
			I += np.real( V*np.exp(2*np.pi*1j*(X*u + Y*v)))

	#Normalize dirty beam
	B /= np.max(B)

	#
	# CLEAN algorithm
	#
	
	Iclean    = np.zeros_like(I)
	residuals = np.copy(I)
	
	if max_iter==None:
		max_iter = int(Npoints/gain)
	
	for ii in range(max_iter):

		#Find maximum peak position
		px,py = np.unravel_index(np.argmax(np.abs(residuals)), residuals.shape)
	
		f = gain*residuals[px,py]
		Iclean[px,py] = f
		
		residuals -= f*np.roll(B, shift=((px-npoints),(py-npoints)), axis=(0,1) )[npoints:-npoints,npoints:-npoints]

		if np.std(residuals) < threshold:
			print(f'Threshold reached after {ii+1} iterations!')
			break

	#Reconvolve with central lobe gaussian
	psf = elliptical_beam_abc(
	X, Y, 
	A=1., 
	Ax=Ax,Axy=Axy,Ay=Ay
	)

	Imap = fftconvolve(Iclean,psf, mode='same') #+ residuals
	Imap /=np.max(Imap)

	return (xb, B.real), (x, I.real), (x,Imap.real)
