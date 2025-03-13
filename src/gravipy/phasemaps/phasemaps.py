import numpy as np
from scipy.fft import fftshift, fftfreq

#COnvolution functions
from scipy.interpolate import RegularGridInterpolator
from astropy.convolution import Gaussian2DKernel
from scipy import signal

#Mathematical functions
from .zernike_polynomials import zernike_polynomial

#Date read tools
import copy
import json
import pickle
import pathlib
from pkg_resources import resource_filename

#Logging tools
import logging
from ..logger.log import log_level_mapping
logging.getLogger("matplotlib").setLevel(logging.WARNING)

#Units
from ..physical_units import units as units

#data structure templates
from .data_structure_templates import *

#Plotting tools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Helper function for templating phasemaps functions
def return_one(args):
	return 1

def return_zero(args):
	return 0

class GravPhaseMaps():
	"""GRAVITY phasemaps class
	"""

	def __init__(self, loglevel='INFO'):
	
		#Create a logger and set log level according to user
		self.logger = logging.getLogger(type(self).__name__)
		self.logger.setLevel(log_level_mapping.get(loglevel, logging.INFO))
		
		# ---------------------------
		# Pre-define class quantities
		# ---------------------------
		
		self.logger.info( 'Phasemap class initialized. No phasemaps loaded.')
		self.logger.info('Either load them using self.load_phasemaps or create them using self.create_phasemaps')
		
		self.resolution = 'LOW'

		self.wave_list = np.array([2.000, 2.037, 2.074, 2.1110000, 2.148, 2.1850002,
                                   2.222, 2.259, 2.296, 2.3330002, 2.370, 2.4070000,
                                   2.444, 2.481    ])

		self.spectral_channels = len(self.wave_list)

		self.l_ref = 2.2 # Reference wavelenght

		#UT related quantities
		self.tel = 'UT'
		self.MFR = 0.6308 # Mode fiber radius (ratio of fiber size to telescope radius)
		self.d1  = 8.0    # Main mirror size in meters
		
		self.stopB = 8.00 # Aperture outer radius
		self.stopS = 0.96 # Aperture inner radius

		self.phasemap_reference_sigma = ( self.l_ref*units.micrometer) / (np.pi * self.MFR * self.d1) #Fiber size in image plane

		self.phasemap_size = 201
		# ------------------
		# Storage variables
		# ------------------

		self.data_folder = 'data/'
		self.zernike_file_2019 = 'zernike_coefficients_2019.json'
		self.zernike_file_2020 = 'zernike_coefficients_2020.json'

		self.zernike_coefficients = None 

		#Storage variable for interpolating function (defaults to zero function)
		self.phasemaps = {
			'GV1' : return_one,
			'GV2' : return_one,
			'GV3' : return_one,
			'GV4' : return_one,
		}
	
		self.phasemaps_normalization = {
			'GV1' : return_one, 
			'GV2' : return_one,
			'GV3' : return_one,
			'GV4' : return_one,
		}
	
		self.phasemaps_amplitude = {
			'GV1' : return_one,
			'GV2' : return_one,
			'GV3' : return_one,
			'GV4' : return_one,
		}

		self.phasemaps_phase = {
			'GV1' : return_zero,
			'GV2' : return_zero,
			'GV3' : return_zero,
			'GV4' : return_zero,
		}

		#Storage variables for phasemap data
		self.phasemaps_data = {
			'GV1' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.complex128),
			'GV2' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.complex128),
			'GV3' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.complex128),
			'GV4' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.complex128),
		}

		self.phasemaps_normalization_data = {
			'GV1' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
			'GV2' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
			'GV3' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
			'GV4' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
		}

		self.phasemaps_amplitude_data = {
			'GV1' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
			'GV2' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
			'GV3' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
			'GV4' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
		}
		
		self.phasemaps_phase_data = {
			'GV1' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
			'GV2' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
			'GV3' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
			'GV4' : np.zeros((self.spectral_channels, self.phasemap_size, self.phasemap_size), dtype=np.float64),
		}

	#=========================
	# Phasemap creation tools
	#=========================

	def get_phase_screen(self, wavelength, beam, 
					  	 include_image_plane_aberrations=True,
						 include_pupil_plane_aberrations=True):
		"""Calculates the phasescreen at a specific wavelength and from a specific beam from the list of zernike coefficients
		"""
	
		if self.zernike_coefficients == None:
			raise ValueError("No zernike coefficients loaded. Load them using self.load_zernike_coefficients('zernike_file.json')")
		
		#Image plane quantities
		Npixels = 1024
		pixels = np.arange(Npixels) - (Npixels/2)

		#Note: Indexing ij is required because of the interpolating function

		dalpha = 1.0*units.mas_to_rad
		x, y  = np.meshgrid(pixels*dalpha, pixels*dalpha, indexing='ij') 
		rho   = np.sqrt(x**2 + y**2)
		theta = np.angle(x + 1j*y)
		
		# Step 1: 
		# Calculate aberrations and fiber profile in image plane

		r = rho/self.phasemap_reference_sigma

		zernike = 0

		if include_image_plane_aberrations:
			zernike += self.zernike_coefficients[beam]['B1m1'] * zernike_polynomial(1, -1, r, theta)
			zernike += self.zernike_coefficients[beam]['B1p1'] * zernike_polynomial(1,  1, r, theta)

			zernike += self.zernike_coefficients[beam]['B2m2'] * zernike_polynomial(2,-2, r, theta)
			zernike += self.zernike_coefficients[beam]['B20']  * zernike_polynomial(2, 0, r, theta)
			zernike += self.zernike_coefficients[beam]['B2p2'] * zernike_polynomial(2, 2, r, theta)

		sigma_fiber     = ( wavelength*units.micrometer) / (np.pi * self.MFR * self.d1)

		fiber_ip = (np.exp(-0.5*(rho/sigma_fiber)**2) * np.exp( (2.*np.pi*1j) * zernike/wavelength ))
		
		# Step 2: 
		# Fourier transform fiber profile to pupil plane

		#Pupil plane quantities
		pupil_pixels = fftshift(fftfreq(Npixels, d=dalpha))*units.micrometer*wavelength
		px, py = np.meshgrid(pupil_pixels, pupil_pixels, indexing='ij')
		pupil_rho   = np.sqrt(px**2 + py**2)
		pupil_theta = np.angle(px + 1j*py)
		
		#Pupil aperture
		pupil = np.logical_and(pupil_rho < (self.stopB/2.), pupil_rho > (self.stopS/2.))

		#Transform fiber to pupil plane
		fiber_pp = fftshift(np.fft.fft2(fftshift(fiber_ip)))

		#Step 3:
		# Calculate aberrations in the pupil plane (apodized pupil function)
		xx = 2.*pupil_rho/self.d1

		zernike = 0

		if include_pupil_plane_aberrations:
			zernike += self.zernike_coefficients[beam]['A00']
			
			zernike += self.zernike_coefficients[beam]['A1m1'] * zernike_polynomial(1, -1, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A1p1'] * zernike_polynomial(1,  1, xx, pupil_theta)
			
			zernike += self.zernike_coefficients[beam]['A2m2'] * zernike_polynomial(2, -2, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A2p2'] * zernike_polynomial(2,  2, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A20']  * zernike_polynomial(2,  0, xx, pupil_theta)
			
			zernike += self.zernike_coefficients[beam]['A3m1'] * zernike_polynomial(3, -1, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A3p1'] * zernike_polynomial(3,  1, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A3m3'] * zernike_polynomial(3, -3, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A3p3'] * zernike_polynomial(3,  3, xx, pupil_theta)
			
			#Note: This is where the convention is wrong! -4 <-> -2 and 4 <-> 2
			zernike += self.zernike_coefficients[beam]['A4m4'] * zernike_polynomial(4, -2, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A4p4'] * zernike_polynomial(4,  2, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A4m2'] * zernike_polynomial(4, -4, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A4p2'] * zernike_polynomial(4,  4, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A40']  * zernike_polynomial(4,  0, xx, pupil_theta)
			
			zernike += self.zernike_coefficients[beam]['A5m1'] * zernike_polynomial(5, -1, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A5p1'] * zernike_polynomial(5,  1, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A5m3'] * zernike_polynomial(5, -3, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A5p3'] * zernike_polynomial(5,  3, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A5m5'] * zernike_polynomial(5, -5, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A5p5'] * zernike_polynomial(5,  5, xx, pupil_theta)
			
			zernike += self.zernike_coefficients[beam]['A6m6'] * zernike_polynomial(6, -6, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A6p6'] * zernike_polynomial(6,  6, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A6m4'] * zernike_polynomial(6, -4, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A6p4'] * zernike_polynomial(6,  4, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A6m2'] * zernike_polynomial(6, -2, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A6p2'] * zernike_polynomial(6,  2, xx, pupil_theta)
			zernike += self.zernike_coefficients[beam]['A60']  * zernike_polynomial(6,  0, xx, pupil_theta)
			
		phase = (2.*np.pi) * zernike/wavelength
		apodized_fiber_pp = pupil * fiber_pp * np.exp(1j*phase)

		# Step 4:
		# Inverse transform back to image plane
		
		complexPsf = np.fft.fftshift(np.fft.fft2(fftshift(apodized_fiber_pp)))	

		# Step 5:
		# Cut complex Psf, normalize and return
		
		map_cut_size = int((self.phasemap_size-1)/2)
		cc = slice(int(Npixels/2)-map_cut_size, int(Npixels/2)+map_cut_size+1)

		complexPsf /= np.abs(complexPsf[cc, cc]).max()

		return x[cc, cc]/units.mas_to_rad, y[cc, cc]/units.mas_to_rad, px[cc, cc], py[cc, cc],  fiber_ip[cc, cc], fiber_pp[cc, cc], complexPsf[cc, cc]

	def load_zernike_coefficients(self,zernikefile):
		""" Loads Zernike coefficients to class from a .json file
		"""
		
		self.zernike_coefficients = copy.deepcopy(template_zernike_table)

		#Load zernike coefficients
		with open(zernikefile) as f:

			file_table = json.load(f)

			#Check that file has the correct formatting
			if not file_table.keys() == template_zernike_table.keys():
				raise ValueError('ERROR: Could not read zernike coefficients from file {}. Make sure the json file follows the correct convention.'.format(zernikefile))
			
			for beam in file_table:
				if not file_table[beam].keys() == template_zernike_list.keys():
					raise ValueError('ERROR: Could not read zernike coefficients from file {}. There seems to be a formating problem with beam {} data'.format(zernikefile,beam))

			#If formatting is correct, load the file 
			for beam in file_table:
				for coefficient in file_table[beam]:
					self.zernike_coefficients[beam][coefficient] = file_table[beam][coefficient]

		return 

	def get_phasemaps_filenames(self, zernikefile, smooth_kernel=15):

		filenames = [

			self.data_folder + f'{zernikefile[:-5]}_smooth_{smooth_kernel}_phasemap_data.p',
			self.data_folder + f'{zernikefile[:-5]}_smooth_{smooth_kernel}_amplitude_data.p',
			self.data_folder + f'{zernikefile[:-5]}_smooth_{smooth_kernel}_phase_data.p',
			self.data_folder + f'{zernikefile[:-5]}_smooth_{smooth_kernel}_normalization_data.p',

			self.data_folder + f'{zernikefile[:-5]}_smooth_{smooth_kernel}_phasemap_interpolator.p',
			self.data_folder + f'{zernikefile[:-5]}_smooth_{smooth_kernel}_amplitude_interpolator.p', 
			self.data_folder + f'{zernikefile[:-5]}_smooth_{smooth_kernel}_phase_interpolator.p',
			self.data_folder + f'{zernikefile[:-5]}_smooth_{smooth_kernel}_normalization_interpolator.p', 
		]

		return [ resource_filename(__name__, f) for f in filenames]

	def create_phasemaps_from_file(self, zernikefile, smooth_kernel=15):
		""" Generates the phasemap interpolating functions based on the zernike coeficients in a .json file.
			The function generates 8 pickle files:
			
			{zernikefile}_smooth_{smooth_kernel}_phasemap_data.p
			{zernikefile}_smooth_{smooth_kernel}_phase_data.p
			{zernikefile}_smooth_{smooth_kernel}_amplitude_data.p 
			{zernikefile}_smooth_{smooth_kernel}_normalization_data.p 
			
			{zernikefile}_smooth_{smooth_kernel}_phasemap_interpolator.p
			{zernikefile}_smooth_{smooth_kernel}_phase_interpolator.p
			{zernikefile}_smooth_{smooth_kernel}_amplitude_interpolator.p 
			{zernikefile}_smooth_{smooth_kernel}_normalization_interpolator.p 
		"""

		#Check if phasemaps already exist
		output_filepaths = self.get_phasemaps_filenames(zernikefile, smooth_kernel)

		for file in output_filepaths:
			if pathlib.Path(file).exists():
				raise ValueError('Phasemaps data files seem to exist already')

		#Load zenikefiles
		zernikefile_path = resource_filename(__name__, self.data_folder + zernikefile)
		self.load_zernike_coefficients(zernikefile_path)

		#Gaussian kernel
		kernel = Gaussian2DKernel(x_stddev=smooth_kernel)

		self.logger.info('Creating phasemaps')

		for beam in self.phasemaps_data:

			for idx, wl in enumerate(self.wave_list):
				
				self.logger.info('Creating phasemap points for beam {} at {} micrometers'.format(beam, wl))
				
				#Get phasemap
				x, y, _, _, _, _, complexPsf = self.get_phase_screen(wl,beam)

				amplitude = np.abs(complexPsf)
				phase     = np.angle(complexPsf, deg=True)

				#Convolve with 2D kernel
				self.phasemaps_data[beam][idx] 				 = signal.convolve2d(complexPsf,   kernel, mode='same')
				self.phasemaps_amplitude_data[beam][idx] 	 = signal.convolve2d(amplitude,    kernel, mode='same')
				self.phasemaps_phase_data[beam][idx] 		 = signal.convolve2d(phase,        kernel, mode='same')
				self.phasemaps_normalization_data[beam][idx] = signal.convolve2d(amplitude**2, kernel, mode='same')

			self.logger.info('Creating interpolating function for phasemap')

			self.phasemaps[beam] 				= RegularGridInterpolator((self.wave_list, x[:,0], y[0,:]), self.phasemaps_data[beam])
			self.phasemaps_amplitude[beam]  	= RegularGridInterpolator((self.wave_list, x[:,0], y[0,:]), self.phasemaps_amplitude_data[beam])
			self.phasemaps_phase[beam]  		= RegularGridInterpolator((self.wave_list, x[:,0], y[0,:]), self.phasemaps_phase_data[beam])
			self.phasemaps_normalization[beam]  = RegularGridInterpolator((self.wave_list, x[:,0], y[0,:]), self.phasemaps_normalization_data[beam])

			#self.amplitude_map[beam] = RegularGridInterpolator((self.wave_list, x[0,:], y[:,0]), np.abs(self.phasemaps_data[beam], dtype=np.float32))

		#Save grid data to file
		self.logger.info('Saving phasemap grids to pickle files')
		pickle.dump( self.phasemaps_data 			   , open( output_filepaths[0], "wb" ) )
		pickle.dump( self.phasemaps_amplitude_data 	   , open( output_filepaths[1], "wb" ) )
		pickle.dump( self.phasemaps_phase_data 		   , open( output_filepaths[2], "wb" ) )
		pickle.dump( self.phasemaps_normalization_data , open( output_filepaths[3], "wb" ) )

		self.logger.info('Phasemap grids saved successeully')

		#Save interpolaring functions to file
		self.logger.info('Saving interpolating functions to pickle files')
	
		pickle.dump( self.phasemaps 			   , open( output_filepaths[4], "wb" ) )
		pickle.dump( self.phasemaps_amplitude 	   , open( output_filepaths[5], "wb" ) )
		pickle.dump( self.phasemaps_phase 		   , open( output_filepaths[6], "wb" ) )
		pickle.dump( self.phasemaps_normalization  , open( output_filepaths[7], "wb" ) )

		self.logger.info('Interpolating functions saved successeully')

		return 

	def create_phasemaps(self, year, smooth_kernel):

		if year==2019:
			self.create_phasemaps_from_file(self.zernike_file_2019, smooth_kernel)
		elif year==2020:
			self.create_phasemaps_from_file(self.zernike_file_2020, smooth_kernel)
		else:
			raise ValueError('ERROR: Phasemaps can only be created for 2020')
		return
	
	#=========================
	# Phasemap loading tools
	#=========================

	def load_phasemaps(self, year, smooth_kernel):

		#Get zernikefile depending on the year
		if year==2019:
			zernikefile = self.zernike_file_2019
		elif year==2020:
			zernikefile = self.zernike_file_2020
		else:
			raise ValueError('ERROR: Phasemaps can only be created for 2020')
		
		input_filepaths = self.get_phasemaps_filenames(zernikefile, smooth_kernel)

		#Load phasemap data
		self.phasemaps_data 			   = pickle.load( open( input_filepaths[0], "rb" ) ) 
		self.phasemaps_amplitude_data 	   = pickle.load( open( input_filepaths[1], "rb" ) ) 
		self.phasemaps_phase_data 		   = pickle.load( open( input_filepaths[2], "rb" ) ) 
		self.phasemaps_normalization_data  = pickle.load( open( input_filepaths[3], "rb" ) ) 

		self.phasemaps 			   		= pickle.load( open( input_filepaths[4], "rb" ) )
		self.phasemaps_amplitude 	   	= pickle.load( open( input_filepaths[5], "rb" ) )
		self.phasemaps_phase 		   	= pickle.load( open( input_filepaths[6], "rb" ) )
		self.phasemaps_normalization  	= pickle.load( open( input_filepaths[7], "rb" ) )

		return

	#=========================
	# Phasemap plotting tools
	#=========================

	def plot_phasemaps(self, wavelength = 2.2 , fiber_fov = 80):
		"""Plot intensity and phase maps for the 4 beams
		"""
		
		#Create figure and setup grid
		fig = plt.figure( figsize=(9,4.5))
		gs = gridspec.GridSpec(figure=fig, ncols=4,nrows=2, height_ratios=[1,1], width_ratios=[1,1,1,1.08])
		axes = gs.subplots()

		#Plotting arguments
		pltargsP = {'cmap': 'twilight_shifted', 'levels': np.linspace(-180, 180, 19, endpoint=True)}
		
		pms = self.phasemaps

		for idx, beam in enumerate(pms):
			
			# Get data
	
			x = pms[beam].grid[1]
			y = pms[beam].grid[2]

			xx, yy = np.meshgrid(x, y)
			zz = pms[beam]((wavelength,xx,yy))
			
			rmap = np.sqrt(xx*xx + yy*yy)
			zz[rmap > fiber_fov] = 0.0 

			# Setup axis
			ax1 = axes[0,idx]
			ax2 = axes[1,idx]
			
			ax1.set_title("{} ({} $\\mu m)$".format(beam, wavelength))

			ax1.set_xticklabels([])
		
			if idx != 0:
				ax1.set_yticklabels([])
				ax2.set_yticklabels([])
			
			if idx == 0:
				ax1.set_ylabel("y (mas)")
				ax2.set_ylabel("y (mas)")

			ax2.set_xlabel("x (mas)")

			scale = 1.05*fiber_fov
			for ax in [ax1,ax2]:
				ax.set_xlim(-scale,scale)
				ax.set_ylim(-scale,scale)

			#Plot
			intensity_plot = ax1.pcolormesh(xx, yy, np.abs(zz)/np.max(np.abs(zz)))
			phase_plot = ax2.contourf(xx, yy, np.angle(zz,deg=True), **pltargsP)

			if idx==3:
				divider1 = make_axes_locatable(ax1)
				cax1 = divider1.append_axes("right", size="5%", pad="3%")
				cbar1 = plt.colorbar(intensity_plot, cax=cax1)

				divider2 = make_axes_locatable(ax2)
				cax2 = divider2.append_axes("right", size="5%", pad="3%")
				cbar2 = plt.colorbar(phase_plot, cax=cax2)

			
			circ = plt.Circle((0,0), radius=fiber_fov, facecolor="None", edgecolor='black', linewidth=0.8)
			ax1.add_artist(circ)

			circ = plt.Circle((0,0), radius=fiber_fov, facecolor="None", edgecolor='black', linewidth=0.8)
			ax2.add_artist(circ)

			ax1.set_aspect(1) 
			ax2.set_aspect(1) 

		plt.subplots_adjust(wspace=0, hspace=0)

		fig.tight_layout()

		return fig, axes