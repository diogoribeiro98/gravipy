import numpy as np
from datetime import datetime

#Parent classes and functions
from ..phasemaps import GravPhaseMaps
from ..data import GravData_scivis
from ..data import InterferometricData

#Units
from ..physical_units import units as units

#Fitting tools
from scipy.optimize import curve_fit
from .models import spectral_visibility
import lmfit
import emcee
import multiprocessing


#Python plotting tools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Specific plotting tools
from ..data.dirty_beam import get_dirty_beam
from ..data.plot_idata import plot_interferometric_data
from ..tools.colors import colors_baseline, colors_closure

#Logging tools
import logging
from ..logger.log import log_level_mapping
logging.getLogger("matplotlib").setLevel(logging.WARNING)

#Input output tools
import os
import h5py

#Define gaussian function to fit to histogram

def gauss(x, *p):
	A, mu, sigma = p
	return A*np.exp(-(x-mu)**2/(2.*sigma**2))

#Alias between beam and telescope
telescope_to_beam = {
	'UT1' : 'GV4',
	'UT2' : 'GV3',
	'UT3' : 'GV2',
	'UT4' : 'GV1',
}

class GraviFit(GravPhaseMaps):
	"""GRAVITY single night fit class
	"""

	def __init__(self,loglevel='INFO'):	
		
		#Create a logger and set log level according to user
		self.logger = logging.getLogger(type(self).__name__)
		self.logger.setLevel(log_level_mapping.get(loglevel, logging.INFO))
		
		#Super constructor
		GravPhaseMaps.__init__(self,loglevel=loglevel)
		
		# ---------------------------
		# Pre-define class quantities
		# ---------------------------
		
		self.class_mode = 'empty' # Can be fits or hdf

		#Status variables
		self.parameters_setup 	= False
		self.fit_performed 		= False
		self.chain_analysed_by_user = False

		#Fitting parameters
		self.params = None
		
		#Fitting helper quantities
		self.field_type = None
		self.nsources 	= None
		
		#Phasemap helper quantities
		self.use_phasemaps = None
		self.phasemap_year = None
		self.phasemap_smoothing_kernel = None

		self.idata = None
		self.flagged_channels	= []

		#
		# Parameters only accessible after a fit is performed 
		# or after data is laoded from hdf file
		#
		# A few notes:
		#
		# - The self.sampler variable is only defined if a fit is performed.
		#
		# - The variables in the second block below are only avaliable if a fit
		#	is performed or if a file is loaded
		# 
		# - Those in the third block are only defined after the fit walkers are
		#	inspected by the user or a file is loaded
		
		#Avaliable after a fit is performed 
		self.sampler = None 

		#Avaliable after a fit is performed or after a file is loaded
		self.fit_weights = None
		self.mcmc_variables = None
		self.max_loglike_parameters = None

		#Avaliable after GraviFit.inspect_walkers is called or a file is loaded
		self.mcmc_chains = None
		self.clean_chain = None
		self.step_cut = None
		self.thin = None
		self.sigma = None
		self.fit_parameters = None

	#========================================
	# Loading functions
	#=========================================

	def load_fits(self, 
			filename , 
			polarization=None,
			flag_channels=[]):

		self.class_mode = 'fits'
		gfit = GravData_scivis(filename)

		self.idata = gfit.get_interferometric_data(
			pol=polarization,
			channel='SC',
			flag_channels=flag_channels
			)
		
		self.flagged_channels = self.flagged_channels
		
		return

	def load_hdf(self, filename):
		
		self.class_mode = 'hdf'
	
		with h5py.File(filename, "r") as f:

			#
			# Load interferometric data
			#

			#Meta data
			grp = f['data']
			filename 	= grp.attrs['filename']
			date_obs	= grp.attrs['date_obs']
			obj 		= grp.attrs['object']
			ra 			= grp.attrs['ra']
			dec			= grp.attrs['dec']
			sobj    	= grp.attrs['sobj']
			sobj_x  	= grp.attrs['sobj_x']
			sobj_y  	= grp.attrs['sobj_y']
			sobj_offx  	= grp.attrs['sobj_offx']
			sobj_offy  	= grp.attrs['sobj_offy']

			#Metrology data
			beams = ['GV1', 'GV2', 'GV3', 'GV4']

			grp = f['data/metrology']

			sobj_metrology_correction_x = { key : grp[key+'_offx'][()] for key in beams }
			sobj_metrology_correction_y = { key : grp[key+'_offy'][()] for key in beams }
			north_angle = {	key : grp[key+'_northangle'][()] for key in beams }

			#Baselines
			grp = f['data/array']
			Bu 		= grp['Bu'][()]
			Bv 		= grp['Bv'][()]
			telpos 	= grp['telpos'][()]

			bl_telescopes = grp['bl_telescopes'][()].astype('U')
			t3_telescopes = grp['t3_telescopes'][()].astype('U')
			
			bl_labels = grp['bl_labels'][()].astype('U')
			t3_labels = grp['t3_labels'][()].astype('U')

			grp = f['data/visibility/wave']
			pol  = str(grp['polarization'][()])
			wave = grp['wave'][()]
			band = grp['band'][()]

			#Visibility data
			grp = f['data/visibility/vis']
			visamp 		= grp['visamp'][()]
			visamp_err 	= grp['visamp_err'][()]
			visphi		= grp['visphi'][()]
			visphi_err 	= grp['visphi_err'][()]
			vis_flag 	= grp['vis_flag'][()]

			grp = f['data/visibility/vis2']
			vis2 		= grp['vis2'][()]
			vis2_err 	= grp['vis2_err'][()]
			vis2_flag 	= grp['vis2_flag'][()]

			grp = f['data/visibility/t3']
			t3amp 		= grp['t3amp'][()]
			t3amp_err 	= grp['t3amp_err'][()]
			t3phi 		= grp['t3phi'][()]
			t3phi_err 	= grp['t3phi_err'][()]
			t3_flag 	= grp['t3_flag'][()]

			grp = f['data/visibility/sf']
			spatial_frequency 	 = grp['spatial_frequency'][()]
			spatial_frequency_t3 = grp['spatial_frequency_t3'][()]

			self.idata = InterferometricData(
						filename=filename,
						date_obs=date_obs,
						object = obj,
						ra	= ra, 	
						dec = dec, 
						sobj   = sobj,
						sobj_x = sobj_x,
						sobj_y = sobj_y,
						sobj_offx = sobj_offx,
						sobj_offy = sobj_offy,
						sobj_metrology_correction_x = sobj_metrology_correction_x,
						sobj_metrology_correction_y = sobj_metrology_correction_y,
						north_angle = north_angle,
						pol=pol,
						Bu=Bu, Bv=Bv, 
						tel_pos=telpos,
						bl_telescopes=bl_telescopes, t3_telescopes=t3_telescopes,
						bl_labels=bl_labels, t3_labels=t3_labels,
						wave=wave, band=band,
						visamp=visamp, visamp_err=visamp_err,
						visphi=visphi, visphi_err=visphi_err,
						vis_flag=vis_flag,
						vis2=vis2, vis2_err=vis2_err, 
						vis2_flag=vis2_flag,
						t3amp=t3amp, t3amp_err=t3amp_err,
						t3phi=t3phi, t3phi_err=t3phi_err,
						t3flag=t3_flag,
						spatial_frequency_as=spatial_frequency,
						spatial_frequency_as_T3=spatial_frequency_t3
					)

			#
 			# Read fit results
			#
			
			grp = f["fit"]
			self.fit_performed = grp.attrs['fit_performed']
			self.parameters_setup = grp.attrs['parameters_setup']
			
			if self.parameters_setup:

				grp_params = f["fit/parameters"]
				
				self.field_type = grp_params.attrs['field_type']
				self.nsources 	= grp_params.attrs['nsources']
				
				self.use_phasemaps = grp_params.attrs['use_phasemaps']
				
				if self.use_phasemaps:
					self.phasemap_year 	= grp_params.attrs['phasemap_year']
					self.phasemap_smoothing_kernel = grp_params.attrs['phasemap_smoothing_kernel']
				
				grp_init = f["fit/parameters/initial"]
				params_dict = {name: grp_init[name][()] for name in grp_init.keys()}
				self.params = self.assemble_parameter_class(params_dict)
			
			if self.fit_performed:

				grp_init = f["fit/parameters/result"]
				params_dict = {name: grp_init[name][()] for name in grp_init.keys()}
				self.fit_parameters = self.assemble_parameter_class(params_dict)

			# Read MCMC results
			grp = f["fit/emcee"]
			self.mcmc_chains = np.zeros((grp.attrs['nsteps'], grp.attrs['nwalkers'], grp.attrs['ndim']))
			self.thin = grp.attrs['thin']
			
			grp_full_chain = f["fit/emcee/full_chain"]
			for idx, param in enumerate(grp_full_chain.keys()):
				self.mcmc_chains[:, :, idx] = grp_full_chain[param][()]
			
			grp_clean_chain = f["fit/emcee/clean_chain"]
			self.step_cut = grp_clean_chain.attrs['step_cut']
			self.sigma = grp_clean_chain.attrs['sigma_cut']
			self.clean_chain = np.zeros((grp_clean_chain[param].shape[0], len(grp_clean_chain.keys())))
			
			self.mcmc_variables = np.zeros(len(grp_clean_chain.keys()), dtype=object)
			for idx, param in enumerate(grp_clean_chain.keys()):
				self.clean_chain[:, idx] = grp_clean_chain[param][()]
				self.mcmc_variables[idx] = param
		return

	#========================================
	# Fitting parameters setup
	#=========================================
	
	def prep_fit_parameters(self,
		
		#Star parameters
		ra_list,
		de_list,
		fr_list,
		star_alpha = 3.0,
		fit_star_pos = True,
		fit_star_fr  = True,
		fit_star_alpha = False,
		
		#SgrA parameters
		sgr_ra = 0.0,
		sgr_de = 0.0,
		sgr_fr = 1.0,
		sgr_alpha = 3.0,
		fit_sgr_pos = True,
		fit_sgr_fr  = True,
		fit_sgr_alpha = True,
		
		#Background parameters
		background_alpha = 3.0,
		background_fr = 0.1,
		fit_background_fr = True,
		fit_background_alpha = True,

		#Field type and fitting model
		field_type = 'star',
		fit_window_stars = None,
		fit_window_sgr = None,

		#Use phasemaps?
		use_phasemaps = True,
		phasemap_year = 2020,
		phasemap_smoothing_kernel = 15
						 ):
		"""Set up fitting parameters for fitting procedure

		Args:
			ra_list (list): list of stars` right ascensions 
			de_list (list): list of stars` declinations
			fr_list (list): list of stars` flux ratios
			star_alpha (float, optional): Spectral index of stars. Defaults to 3.0.
			fit_star_pos (bool, optional): If True, star positions are fitted. Defaults to True.
			fit_star_fr (bool, optional): If True, star fluxes (magnitudes) are fitted. Defaults to True.
			fit_star_alpha (bool, optional): If True, spectral index of stars is fitted. Defaults to False.
			
			sgr_ra (float, optional): Right ascension of Sgr A*. Defaults to 0.0.
			sgr_de (float, optional): Declination of Sgr A*. Defaults to 0.0.
			sgr_fr (float, optional): Flux ratio of SgrA*. Defaults to 1.0.
			sgr_alpha (float, optional): Spectral index of Sgr A*. Defaults to 3.0.
			fit_sgr_pos (bool, optional): If True, position of Sgr A* is fitted. Defaults to True.
			fit_sgr_fr (bool, optional): If True, flux ratio (magnitude) of Sgr A* if fitted. Defaults to True.
			fit_sgr_alpha (bool, optional): If True, spectral ratio of Sgr A* if fitted. Defaults to True.
			
			background_alpha (int, optional): Spectral index of background. Defaults to 3.0.
			background_fr (float, optional): Flux ratio of background with respect to first source. Defaults to 0.1.
			fit_background_fr (bool, optional): If True, background flux is fitted. Defaults to True.
			fit_background_alpha (bool, optional): If True, spectral index of background is fitted. Defaults to True.

			field_type (str, optional): Type of field to fit. Allowed values are 'star' and 'sgra'. Defaults to 'star'.
			fit_window_stars (_type_, optional): Size of flat prior around initial position of stars. If None is give, defaults to 5 miliarcseconds.
			fit_window_sgr (_type_, optional): Size of flat prior around Sgr A* position.If None is given, defaults to 5 miliarcseconds.
		"""
		
		#Check if the list of RA, Dec and Flux all have the same length
		if not all(len(lst) == len(ra_list) for lst in [de_list, fr_list]):
			raise  ValueError('RA, Dec and Flux lists must have the same length!')
		else:
			nsource = len(ra_list)

		# -----------------------------------------------------------------
		# Setup phasemaps if requested
		# -----------------------------------------------------------------
		
		self.use_phasemaps = use_phasemaps
		
		if use_phasemaps:
			
			self.phasemap_year = phasemap_year
			self.phasemap_smoothing_kernel = phasemap_smoothing_kernel

			self.load_phasemaps(year = phasemap_year, smooth_kernel=phasemap_smoothing_kernel)

		# -----------------------------------------------------------------
		# Create a different set of parameters depending on the field type
		# -----------------------------------------------------------------

		if field_type=='star':
			
			self.logger.info( 'Setting up star field fit')

			if nsource==0:
				raise ValueError('Field type is star but no initial guess was given. Give at least one ra,dec,flux set of values.')

			#Setup class and create list of sources
			self.field_type = 'star' 
			self.nsources = nsource
			
			# Parameters for stars 
			# number of parameters = (3n-1)+1
			star_fit_parameters = {
				'stars_alpha' : [ star_alpha , -10, 10, fit_star_alpha]
			}

			#Check the fit_window_argument
			if fit_window_stars == None:
				fit_window = np.ones(nsource)*5.0
			elif isinstance(fit_window_stars,float):
				fit_window = np.ones(nsource)*fit_window_stars
			elif len(fit_window_stars) == nsource:
				fit_window = fit_window_stars
			else:
				raise ValueError('fit_window_stars must be either a float or array with the size of star sources!')

			for idx in range(nsource):
				
				star_fit_parameters[f'source_{idx}_ra']   = [ ra_list[idx],  ra_list[idx] - fit_window[idx],  ra_list[idx] + fit_window[idx], fit_star_pos ]
				star_fit_parameters[f'source_{idx}_dec']  = [ de_list[idx],  de_list[idx] - fit_window[idx],  ra_list[idx] + fit_window[idx], fit_star_pos ]
				
				#If first star, fix the flux
				if idx==0:
					star_fit_parameters[f'source_{idx}_dmag'] = [ 0.0, -6.0 , 6.0, False ]
				else:
					star_fit_parameters[f'source_{idx}_dmag'] = [ -2.5*np.log10(fr_list[idx]/fr_list[0]), -6.0, 6.0, fit_star_fr ]

			#Fitting parameters for background
			# number of parameters = 2
			background_fit_parameters = {
				'background_flux' : [background_fr   ,   0.0, 10.0, fit_background_fr   ],
				'background_alpha': [background_alpha, -10.0, 10.0, fit_background_alpha]
			}		

			#Create a dictionary with all the parameters to fit and assemble parameter class
			all_fitting_parameters = {}
			all_fitting_parameters.update(star_fit_parameters) 
			all_fitting_parameters.update(background_fit_parameters)

			self.params = self.assemble_parameter_class(all_fitting_parameters)
			
		elif field_type=='sgra':

			self.logger.info( 'Setting up sgra field fit' )

			if nsource==0:
				self.logger.info('No sources exept sgra. Fitting single source.')

				#Setup class and create list of sources
				self.field_type = 'sgra' 
				self.nsources = 0
	
				#Check fitting area
				if fit_window_sgr == None:
					fit_window = 5.0
				elif isinstance(fit_window_sgr,float):
					fit_window = fit_window_sgr
				else:
					raise ValueError('fit_window_sgr must be a float!')

				sgra_fit_parameters = {
					'sgra_ra' 	 : [sgr_ra	 	, -fit_window	 , fit_window	 , fit_sgr_pos  ],
					'sgra_dec' 	 : [sgr_de	 	, -fit_window	 , fit_window	 , fit_sgr_pos  ],
					'sgra_dmag'  : [0.0			, -6.0			 , 6.0			 , False        ],
					'sgra_alpha' : [sgr_alpha	, -10.0	 		 , 10.0			 , fit_sgr_alpha],
				}

				#Fitting parameters for background
				# number of parameters = 2
				background_fit_parameters = {
					'background_flux' : [background_fr   ,  0.0, 10.0, fit_background_fr   ],
					'background_alpha': [background_alpha, -10.0, 10.0, fit_background_alpha]
				}	

				all_fitting_parameters = {}
				all_fitting_parameters.update(sgra_fit_parameters) 
				all_fitting_parameters.update(background_fit_parameters)

				self.params = self.assemble_parameter_class(all_fitting_parameters)

			else:

				#Setup class and create list of sources
				self.field_type = 'sgra' 
				self.nsources = nsource
				
				#Fitting parameters for stars 
				# number of parameters = (3n-1)+1
				star_fit_parameters = {
					'stars_alpha' : [ star_alpha , -10, 10, fit_star_alpha]
				}

				#Check the fit_window_argument
				if fit_window_stars == None:
					fit_window = np.ones(nsource)*5.0
				elif isinstance(fit_window_stars,float):
					fit_window = np.ones(nsource)*fit_window_stars
				elif len(fit_window_stars) == nsource:
					fit_window = fit_window_stars
				else:
					raise ValueError('fit_window_stars must be either a float or array with the size of star sources!')

				for idx in range(nsource):
					
					star_fit_parameters[f'source_{idx}_ra']   = [ ra_list[idx],  ra_list[idx] - fit_window[idx],  ra_list[idx] + fit_window[idx], fit_star_pos ]
					star_fit_parameters[f'source_{idx}_dec']  = [ de_list[idx],  de_list[idx] - fit_window[idx],  de_list[idx] + fit_window[idx], fit_star_pos ]
					
					#If first star, fix the flux
					if idx==0:
						star_fit_parameters[f'source_{idx}_dmag'] = [ 0.0, -6.0,  6.0 , False ]
					else:
						star_fit_parameters[f'source_{idx}_dmag'] = [ -2.5*np.log10(fr_list[idx]/fr_list[0]), -6.0 ,  6.0, fit_star_fr ]

				#Check fitting area
				if fit_window_sgr == None:
					fit_window = 5.0
				elif isinstance(fit_window_sgr,float):
					fit_window = fit_window_sgr
				else:
					raise ValueError('fit_window_sgr must be a float!')

				sgra_fit_parameters = {
					'sgra_ra' 	 : [sgr_ra	 			, -fit_window	 , fit_window	 , fit_sgr_pos  ],
					'sgra_dec' 	 : [sgr_de	 			, -fit_window	 , fit_window	 , fit_sgr_pos  ],
					'sgra_dmag'  : [-2.5*np.log10(sgr_fr/fr_list[0]), -6.0, 6.0			 , fit_sgr_fr   ],
					'sgra_alpha' : [sgr_alpha			, -10.0	 		 , 10.0			 , fit_sgr_alpha],
				}

				#Fitting parameters for background
				# number of parameters = 2
				background_fit_parameters = {
					'background_flux' : [background_fr   ,  0.0, 10.0, fit_background_fr   ],
					'background_alpha': [background_alpha, -10.0, 10.0, fit_background_alpha]
				}	

				#Create a dictionary with all the parameters to fit
				all_fitting_parameters = {}
				all_fitting_parameters.update(star_fit_parameters) 
				all_fitting_parameters.update(sgra_fit_parameters) 
				all_fitting_parameters.update(background_fit_parameters)

				self.params = self.assemble_parameter_class(all_fitting_parameters)
				
		else:
			raise ValueError('Field type not recognized. Field type must be "stars" or "sgra".')

		self.parameters_setup = True

		return

	def assemble_parameter_class(self, parameter_dictionary):
		""" Generates a lmfit.Parameters class from a parameter dictionary.
			The parameter dictionary should be of the form

			dict = {'parameter name' : [value, min_value, max_value, vary]}

			where min_value and max_value represent the bounds on the parameter
			and vary wether or not the parameter should be varied during the fit		
		"""

		params = lmfit.Parameters()
		
		for name in parameter_dictionary:
			
			params.add(
				name, 
				value = parameter_dictionary[name][0],
				min   = parameter_dictionary[name][1],
				max   = parameter_dictionary[name][2],
				vary  = bool(parameter_dictionary[name][3]),
				)

			#If the list is longer it contains the standard deviation of the parameter
			if len(parameter_dictionary[name]) == 5:
				params[name].stderr = parameter_dictionary[name][4]
		
		return params
	
	#====================
	# Visibility model 
	#===================

	@staticmethod
	def nsource_visibility(
			uv_coordinates,
			sources,
			background,
			l_list,
			dl_list,
			reference_l0=2.2,
			use_phasemaps=False,
			phase_maps = None,
			amplitude_maps = None,
			normalization_maps = None,
			north_angle = [0.0,0.0],
			met_offx = [0.0, 0.0],
			met_offy = [0.0, 0.0],
			):
		
		u,v = uv_coordinates

		#Storage variable for opd
		opd = np.zeros_like(l_list)

		#Calculate things differently if using phasemaps or not
		if not use_phasemaps:
	
			#Storage variables
			visibility = np.zeros_like(l_list, dtype=np.complex128)
			normalization = np.zeros_like(l_list)
			
			#for x, y, flux, alpha in sources:
			for src in sources:

				#Get position, flux and spectral index for each source
				x, y, flux, alpha = src	

				#Calculate optical path difference from position on sky and visibility
				opd.fill((u*x + v*y)*units.mas_to_rad)

				#Add source visibility to nsource one
				visibility += flux*spectral_visibility(opd, alpha, l_list, dl_list, reference_l0)
				normalization += flux*spectral_visibility(0, alpha, l_list, dl_list, reference_l0)
			
			#Add background to normalization
			flux, alpha = background
			normalization += flux*spectral_visibility(0, alpha, l_list, dl_list,reference_l0)
			
			#Calculate spatial frequencies
			sf = np.sqrt(u**2+v**2)/l_list*units.as_to_rad

			return sf, visibility/normalization

		else:

			#Storage variables
			visibility = np.zeros_like(l_list, dtype=np.complex128)
			normalization_i = np.zeros_like(l_list)
			normalization_j = np.zeros_like(l_list)
			
			#Get phasemaps
			phi_i, phi_j = phase_maps
			Ai, Aj = amplitude_maps
			Li, Lj = normalization_maps

			#Helper vectors to vectorize operations
			xi_list = np.zeros_like(l_list)
			yi_list = np.zeros_like(l_list)

			xj_list = np.zeros_like(l_list)
			yj_list = np.zeros_like(l_list)

			#Metrology offset and rotation corrections
			offset_i = np.array([met_offx[0], met_offy[0]])
			offset_j = np.array([met_offx[1], met_offy[1]])
			
			Ri = np.array([	[np.cos(north_angle[0]), -np.sin(north_angle[0])], 
				  			[np.sin(north_angle[0]),  np.cos(north_angle[0])]])

			Rj = np.array([	[np.cos(north_angle[1]), -np.sin(north_angle[1])], 
				  			[np.sin(north_angle[1]),  np.cos(north_angle[1])]])
			
			for src in sources:

				#Get position, flux and spectral index for each source
				x, y, flux, alpha = src
			
				#Offset phasemap evaluation coordinates according to metrology

				xi,yi = np.matmul(Ri, [x,y] + offset_i)
				xj,yj = np.matmul(Rj, [x,y] + offset_j)

				#Calculate optical path difference from position on sky and visibility
				opd.fill((u*x + v*y)*units.mas_to_rad)
				
				xi_list.fill(xi)
				yi_list.fill(yi)
				
				xj_list.fill(xj)
				yj_list.fill(yj)

				#Correct with phasemaps
				arg_i = (l_list,xi_list,yi_list)
				arg_j = (l_list,xj_list,yj_list)

				opd -= ( phi_i(arg_i) - phi_j(arg_j))*l_list/360. 
				Lij  =  Ai(arg_j)*Aj(arg_j)

				#Calculate visibility
				visibility += Lij*flux*spectral_visibility(opd, alpha, l_list, dl_list, reference_l0)
				
				# Normalization terms
				sv0 = spectral_visibility(0, alpha, l_list, dl_list, reference_l0)
				
				normalization_i += Li(arg_i) * flux * sv0
				normalization_j += Lj(arg_j) * flux * sv0
				
		 	#Add background to normalization
			flux, alpha = background
			normalization_i += flux*spectral_visibility(0, alpha, l_list, dl_list,reference_l0)
			normalization_j += flux*spectral_visibility(0, alpha, l_list, dl_list,reference_l0)
			
			#Calculate spatial frequencies
			sf = np.sqrt(u**2+v**2)/l_list*units.as_to_rad
			
			return sf, visibility/np.sqrt(normalization_i*normalization_j)
			
	def get_visibility_model(self, params, use_phasemaps=False):
		
		#Get baseline info from datafile and create template visibility model
		idata = self.idata

		visibility_model = dict((str(name),None) for name in idata.bl_labels)	

		#Get sources from parameters
		sources, background = GraviFit.get_sources_and_background(params.valuesdict(), self.field_type, self.nsources )

		for idx, (telescopes, label) in enumerate(zip(idata.bl_telescopes, idata.bl_labels)):
			
			ucoord = idata.Bu[idx]/units.micrometer
			vcoord = idata.Bv[idx]/units.micrometer

			if use_phasemaps:

				phase_maps = [self.phasemaps_phase[telescope_to_beam[telescopes[0]]],
							  self.phasemaps_phase[telescope_to_beam[telescopes[1]]]]
						
				amplitude_maps = [	self.phasemaps_amplitude[telescope_to_beam[telescopes[0]]],
									self.phasemaps_amplitude[telescope_to_beam[telescopes[1]]]]
						
				normalization_maps = [	self.phasemaps_normalization[telescope_to_beam[telescopes[0]]],
										self.phasemaps_normalization[telescope_to_beam[telescopes[1]]]]

				met_offx = [ idata.sobj_metrology_correction_x[telescope_to_beam[telescopes[0]]],
							 idata.sobj_metrology_correction_x[telescope_to_beam[telescopes[1]]]]

				met_offy = [ idata.sobj_metrology_correction_y[telescope_to_beam[telescopes[0]]],
							 idata.sobj_metrology_correction_y[telescope_to_beam[telescopes[1]]]]

				nangle = [	idata.north_angle[telescope_to_beam[telescopes[0]]],
							idata.north_angle[telescope_to_beam[telescopes[1]]]]

				phasemap_args = {
					'phase_maps': phase_maps,
					'amplitude_maps': amplitude_maps,
					'normalization_maps': normalization_maps,
					'north_angle': nangle,
					'met_offx': met_offx,
					'met_offy': met_offy
				}

			else:
				phasemap_args = {}

			model = GraviFit.nsource_visibility(
			uv_coordinates= [ucoord,vcoord],
			sources=sources,
			background=background,
			l_list=idata.wave,
			dl_list=idata.band,
			use_phasemaps=use_phasemaps,
			**phasemap_args
			)

			visibility_model[label] = model
				
		return visibility_model
		
	#========================================
	# Static methods for multithread fitting
	#========================================
	
	@staticmethod
	def get_sources_and_background(parameter_value_dictionary, field_type, nsource):
		"""Returns the source and background arrays associated with the given
		parameter dictionary
		"""

		if field_type=='star':

			sources = np.zeros((nsource, 4)) 	# [ra,dec,flux,alpha] for each source
			background = np.zeros(2)			# [flux,alpha]

			stars_alpha = parameter_value_dictionary['stars_alpha']
			background_flux = parameter_value_dictionary['background_flux']
			background_alpha = parameter_value_dictionary['background_alpha']

			for idx in range(nsource):
				sources[idx, 0] = parameter_value_dictionary[f'source_{idx}_ra']
				sources[idx, 1] = parameter_value_dictionary[f'source_{idx}_dec']
				sources[idx, 2] = 10**(-0.4*parameter_value_dictionary[f'source_{idx}_dmag'])
				sources[idx, 3] = stars_alpha  # Same for all sources
		
			background[:] = [background_flux, background_alpha]  

		elif field_type=='sgra':
			
			#If sgra is the only source in the field
			if nsource==0:
				
				sources = np.zeros((1, 4))  # [ra,dec,flux,alpha] for SgrA*
				background = np.zeros(2)	# [flux,alpha]

				background_flux = parameter_value_dictionary['background_flux']
				background_alpha = parameter_value_dictionary['background_alpha']

				sources[0, 0] = parameter_value_dictionary['sgra_ra']
				sources[0, 1] = parameter_value_dictionary['sgra_dec']
				sources[0, 2] = 10**(-0.4*parameter_value_dictionary['sgra_dmag'])
				sources[0, 3] = parameter_value_dictionary['sgra_alpha']

				background[:] = [background_flux, background_alpha] 

			else:

				sources = np.zeros((nsource + 1, 4)) # [ra,dec,flux,alpha] for each source and for Sgr A*
				background = np.zeros(2)			 # [flux,alpha]

				stars_alpha = parameter_value_dictionary['stars_alpha']
				background_flux = parameter_value_dictionary['background_flux']
				background_alpha = parameter_value_dictionary['background_alpha']
			
				for idx in range(nsource):
					sources[idx, 0] = parameter_value_dictionary[f'source_{idx}_ra']
					sources[idx, 1] = parameter_value_dictionary[f'source_{idx}_dec']
					sources[idx, 2] = 10**(-0.4*parameter_value_dictionary[f'source_{idx}_dmag'])
					sources[idx, 3] = stars_alpha  # Same for all sources
			
				sources[-1, 0] = parameter_value_dictionary['sgra_ra']
				sources[-1, 1] = parameter_value_dictionary['sgra_dec']
				sources[-1, 2] = 10**(-0.4*parameter_value_dictionary['sgra_dmag'])
				sources[-1, 3] = parameter_value_dictionary['sgra_alpha'] 

				background[:] = [background_flux, background_alpha]  

		return sources, background

	@staticmethod
	def log_likelihood(theta, theta_names):
		"""
		Loglikelihood function for emcee sampler.
		
		The functions is made to run in parallel and needs
		access to global variables. As such it should only be
		called from within an open pool of workers with the proper
		emcee worker initializer (see GravMfit.run_emcee_fit) function

		"""
		
		#
		# Check if walker is outside of prior bounds
		#
		
		global uniform_prior_bounds
		global parameter_value_dictionary

		theta_dictionary = dict(zip(theta_names, theta))
		
		for key in parameter_value_dictionary:
			parameter_value_dictionary[key] = theta_dictionary.get(key, parameter_value_dictionary[key])

		for key, value in theta_dictionary.items():
			min_bound, max_bound = uniform_prior_bounds[key]
			if not (min_bound <= value <= max_bound):
				return -np.inf  
			
		#
		# Calculate visibility model
		#

		#Fetch sources
		global field_type
		global nsources

		sources, background = GraviFit.get_sources_and_background(
			parameter_value_dictionary=parameter_value_dictionary,
			field_type=field_type,
			nsource=nsources)

		#Get model for all baselines
		global idata 
		global visibility_model 	
		
		global use_phasemaps
		global phasemaps_phase, phasemaps_amplitude
		global phasemaps_normalization
		global north_angle
		global metrology_offx, metrology_offy

		for idx, (telescopes, label) in enumerate(zip(idata.bl_telescopes, idata.bl_labels)):
			
			ucoord = idata.Bu[idx]/units.micrometer
			vcoord = idata.Bv[idx]/units.micrometer

			if use_phasemaps:

				phase_maps = [phasemaps_phase[telescope_to_beam[telescopes[0]]],
							  phasemaps_phase[telescope_to_beam[telescopes[1]]]]
						
				amplitude_maps = [	phasemaps_amplitude[telescope_to_beam[telescopes[0]]],
									phasemaps_amplitude[telescope_to_beam[telescopes[1]]]]
						
				normalization_maps = [	phasemaps_normalization[telescope_to_beam[telescopes[0]]],
										phasemaps_normalization[telescope_to_beam[telescopes[1]]]]

				met_offx = [ metrology_offx[telescope_to_beam[telescopes[0]]],
							 metrology_offx[telescope_to_beam[telescopes[1]]]]

				met_offy = [ metrology_offy[telescope_to_beam[telescopes[0]]],
							 metrology_offy[telescope_to_beam[telescopes[1]]]]

				nangle = [	north_angle[telescope_to_beam[telescopes[0]]],
							north_angle[telescope_to_beam[telescopes[1]]]]

				phasemap_args = {
					'phase_maps': phase_maps,
					'amplitude_maps': amplitude_maps,
					'normalization_maps': normalization_maps,
					'north_angle': nangle,
					'met_offx': met_offx,
					'met_offy': met_offy
				}

			else:
				phasemap_args = {}

		
			model = GraviFit.nsource_visibility(
			uv_coordinates= [ucoord,vcoord],
			sources=sources,
			background=background,
			l_list=idata.wave,
			dl_list=idata.band,
			use_phasemaps=use_phasemaps,
			**phasemap_args
			)

			visibility_model[label] = model
				

		#
		# Calculate residuals
		#

		global weights

		residual_sum = 0.0

		for idx, label in enumerate(visibility_model):
			
			# Model quantities
			visamp_model = np.abs(visibility_model[label][1])
			visphi_model = np.angle(visibility_model[label][1], deg=True)
			vis2_model = visamp_model**2

			# Data pairs for amplitude, phase, and squared residuals (only P1 implemented at the moment)
			amp_data, amp_err = idata.visamp[idx], idata.visamp_err[idx]
			phi_data, phi_err = idata.visphi[idx], idata.visphi_err[idx]
			v2_data , v2_err  = idata.vis2[idx]  , idata.vis2_err[idx]  

			if weights[0] != 0.0:
				residuals_amp = ((visamp_model - amp_data)/amp_err )**2
				residual_sum += np.nansum(residuals_amp)*weights[0]
			
			if weights[1] != 0.0:
				residuals_phi = ((visphi_model - phi_data)/phi_err )**2
				residual_sum += np.nansum(residuals_phi)*weights[1]

			if weights[2] != 0.0:
				residuals_vis2 = ((vis2_model - v2_data)/v2_err )**2
				residual_sum  += np.nansum(residuals_vis2)*weights[2]
		
		if weights[3] != 0.0:
			
			#Closure phases
			c1 = np.angle(visibility_model['UT4-3'][1]) + np.angle(visibility_model['UT3-2'][1]) - np.angle(visibility_model['UT4-2'][1])
			c2 = np.angle(visibility_model['UT4-3'][1]) + np.angle(visibility_model['UT3-1'][1]) - np.angle(visibility_model['UT4-1'][1])
			c3 = np.angle(visibility_model['UT4-2'][1]) + np.angle(visibility_model['UT2-1'][1]) - np.angle(visibility_model['UT4-1'][1])
			c4 = np.angle(visibility_model['UT3-2'][1]) + np.angle(visibility_model['UT2-1'][1]) - np.angle(visibility_model['UT3-1'][1])

			cp = np.array([c1,c2,c3,c4])*180/np.pi
			
			for idx in range(4):

				cphase_data, cphase_err = idata.t3phi[idx], idata.t3phi_err[idx]
				
				residuals_t3 = ((cp[idx] - cphase_data)/cphase_err )**2	
				residual_sum  += np.nansum(residuals_t3)*weights[3]
		
		log_like = -0.5*residual_sum

		return log_like 

	def run_mcmc_fit(
			self,
			nwalkers=50, 
			steps=100, 
			nthreads=1, 
			initial_spread = 0.5, 
			fit_weights=[1.0, 1.0, 0.0, 0.0]):
		"""Run monte carlo fitting

		Args:
			nwalkers (int, optional): _description_. Defaults to 50.
			steps (int, optional): _description_. Defaults to 100.
			nthreads (int, optional): _description_. Defaults to 1.
			initial_spread (float, optional): _description_. Defaults to 0.5.
			fit_weights (list, optional): _description_. Defaults to [1.0, 1.0, 0.0, 0.0].

		Returns:
			_type_: _description_
		"""

		#Check that parameters are setup
		if not self.parameters_setup:
			raise ValueError('Fitting parameters are not setup. Run the `GraviFit.prep_fit_parameters` function')

		#
		# Setup walkers for emcee
		#

		#Initial spread cannot be larger than 1
		if initial_spread >= 1:
			raise ValueError('initial_spread cannot be larger than 1.')


		#Get parameter names to fit
		parameters_to_fit = [ p for p in self.params if self.params[p].vary==True]
		initial_values    = [ self.params[p].value for p in parameters_to_fit] 
		param_range  	  = [ np.abs(self.params[p].max-self.params[p].min)/2 for p in parameters_to_fit] 

		#Setup initial state of walkers
		ndim = len(parameters_to_fit)

		initial_emcee_state = np.array([
		[np.clip(
			initial_values[idx] + initial_spread * np.random.uniform(-param_range[idx], param_range[idx]), 
			 self.params[parameters_to_fit[idx]].min, 
			 self.params[parameters_to_fit[idx]].max) 
	 	for idx in range(ndim)] 
		for _ in range(nwalkers)
		])

		#Setup number of threads
		n_cores = nthreads
		if nthreads > multiprocessing.cpu_count():
			raise ValueError(f'nthreads ({nthreads}) cannot be larger than cpu count on your machine ({multiprocessing.cpu_count()})')
		
		#
		# Setup function for each thread (global variables setup)
		#

		def emcee_worker_init():

			#List global variables
			global uniform_prior_bounds
			global field_type
			global nsources
			global parameter_value_dictionary
			global idata 
			global visibility_model 	
			global use_phasemaps
			global phasemaps_phase, phasemaps_amplitude
			global phasemaps_normalization
			global north_angle
			global metrology_offx, metrology_offy
			global weights

			#Setup values

			uniform_prior_bounds = {
				name: (param.min, param.max)
				for name, param in self.params.items()
				if param.min is not None and param.max is not None
				}

			field_type = self.field_type
			nsources = self.nsources
			
			parameter_value_dictionary = self.params.valuesdict()

			idata = self.idata
	
			visibility_model = dict((str(name),None) for name in idata.bl_labels)

			use_phasemaps 			= self.use_phasemaps
			phasemaps_phase 		= self.phasemaps_phase
			phasemaps_amplitude 	= self.phasemaps_amplitude
			phasemaps_normalization = self.phasemaps_normalization

			north_angle 	= idata.north_angle
			metrology_offx 	= idata.sobj_metrology_correction_x
			metrology_offy 	= idata.sobj_metrology_correction_y
			
			weights = fit_weights

			return			

		#
		# Perform parallel fit
		#

		log_likelihood = GraviFit.log_likelihood
		
		with multiprocessing.Pool(processes=n_cores, initializer=emcee_worker_init) as pool:
	
			sampler = emcee.EnsembleSampler(
				nwalkers=nwalkers, 
				ndim=ndim, 
				log_prob_fn=log_likelihood, 
				pool=pool,
				args=(parameters_to_fit,),
				moves=[
					(emcee.moves.DEMove(), 0.8),
					(emcee.moves.DESnookerMove(), 0.1),
					(emcee.moves.StretchMove(), 0.1)
				] 
				)
			
			sampler.run_mcmc(
				initial_state=initial_emcee_state, 
				nsteps=steps, 
				progress=True)

		#
		# Post-process samples
		#

		#Get the maximum loglikelihood parameters
		samples = sampler.get_chain(flat=True) 
		log_probs = sampler.get_log_prob(flat=True)
		max_likelihood_sample = samples[log_probs.argmax()]

		#Overwrite parameter values
		max_loglike_parameters = self.params.copy()

		for idx, elem in enumerate(parameters_to_fit):
			max_loglike_parameters[elem].value  = max_likelihood_sample[idx]

		#Setup class variables associated with fit

		self.fit_performed 		= True
		self.chain_analysed_by_user = False

		self.sampler = sampler

		self.fit_weights = fit_weights
		self.mcmc_variables = parameters_to_fit
		self.max_loglike_parameters = max_loglike_parameters
		
		return 
	
	#========================================
	# mcmc analysis tools
	#=========================================

	def estimate_chain_percentiles(
		self,
		chain,
		percentiles=[5, 50, 95]
		):
		"""Estimates the percentiles of a flat chain of samples.

		Args:
			chain (np.array): Flat samples with dimenson (samples,ndim) 
			percentiles (list, optional): Percentile list in the form [lower,mid, higher]. Defaults to [5,50,95], approximately corresponding to 2 sigma levels.

		Returns:
			(q_low, mu,q_high): lower percentile, median and upper percentile
		"""
		
		ndim = chain.shape[1]

		#Storage variables
		q_low 	= np.empty(ndim)
		mu 		= np.empty(ndim)
		q_high 	= np.empty(ndim)

		#Calculate the histogram for each of the variables
		for idx,var in enumerate(chain.T):
			hist, bins = np.histogram(var, bins='fd')
			x = (bins[:-1] + bins[1:])/2
			l, m, h = np.percentile(x,percentiles, weights=hist, method='inverted_cdf')
			q_low[idx], mu[idx], q_high[idx] = l,m,h

		return (q_low, mu,q_high)

	def get_gaussian_chain_fit(self,chain):
		"""
		Give a chain chain of samples with size (samples,ndim) returns the gaussian fit 
		to each one of the dimensions.

		Args:
			chain (np.array): Flat samples with dimenson (samples,ndim) 

		Returns:
			mu_list, sigma_list: fit result arrays with dimension (ndim).
		"""
		
		#Number of variables
		ndim = chain.shape[1]

		#Storage variables		
		mu_list = np.empty(ndim)
		sigma_list = np.empty(ndim)

		#Initial guess for fit
		(q_low, mu, q_high) = self.estimate_chain_percentiles(chain)
		
		#Evaluate gaussian fit to clean chain
		for idx in range(ndim):
			samples = chain[:,idx]

			hist, bins = np.histogram(samples, bins='fd',density=True)
			
			#Fit a gaussian to the filtered points
			x = (bins[:-1] + bins[1:])/2

			p0 = [np.max(hist), mu[idx], q_high[idx]-q_low[idx]]
			(A, m, s), _ = curve_fit(
				gauss, 
				x, hist, 
				p0=p0, 
				bounds=([0,-np.inf,0], [np.inf, np.inf, np.inf]))
			
			mu_list[idx] = m
			sigma_list[idx] = s

		return mu_list, sigma_list

	def get_clean_chain(
			self,
			chain,
			lower_limits,
			upper_limits):
		"""Given a flat chain of samples of dimension (nsamples,ndim), returns the 
		filtered out chain according to the user provided lower and upper limits for 
		each dimension.

		Args:
			chain (np.array): Flat samples with dimenson (samples,ndim) 
			lower_limits (array): One dimensonla array with lower limits for variables. Must have size (ndim)
			upper_limits (array): One dimensonla array with upper limits for variables. Must have size (ndim)

		Returns:
			flat_chain: Returns the filtered chain
		"""
		mask = np.all((chain >= lower_limits) & (chain <= upper_limits), axis=1)
		return chain[mask]

	def inspect_walkers(
			self,
			step_cut, 
			thin, 
			sigma = 5
			):

		if not self.fit_performed:
			raise ValueError('No fit has been performed.')

		if self.class_mode=='fits':

			#Get full chain
			chain 		= self.sampler.get_chain(discard=0,  thin=1, flat=False) 
			flat_chain  = self.sampler.get_chain(discard=0,  thin=1, flat=True) 

			nsteps, nwalkers, ndim = chain.shape
		
			#Get filtered chain adn estimate percentiles
			flat_filt_chain = self.sampler.get_chain(discard=step_cut, thin=thin, flat=True) 
		
			#Evaluate gaussian fit to clean chain adn clean chain
			mu_list, sigma_list = self.get_gaussian_chain_fit(flat_filt_chain)
			flat_filt_chain = self.get_clean_chain(flat_filt_chain, mu_list-sigma*sigma_list, mu_list+sigma*sigma_list)

		elif self.class_mode=='hdf':
			
			chain = self.mcmc_chains
			nsteps, nwalkers, ndim = chain.shape
			flat_chain = chain.reshape((nsteps*nwalkers,ndim))
			
			flat_filt_chain = self.mcmc_chains[step_cut::thin,:,:]
			nsteps_filt, nwalkers, ndim = flat_filt_chain.shape
			flat_filt_chain = flat_filt_chain.reshape((nsteps_filt*nwalkers,ndim))
			
			mu_list, sigma_list = self.get_gaussian_chain_fit(flat_filt_chain)
			
			flat_filt_chain = self.get_clean_chain(flat_filt_chain, mu_list-sigma*sigma_list, mu_list+sigma*sigma_list)

		#Create figure and setup plot
		fig = plt.figure(figsize=(8, 2*ndim), dpi=100)

		subfig_left, subfig_right = fig.subfigures(1, 2, wspace=0.1, width_ratios=[3, 1])  
		axes_left  = subfig_left.subplots(ndim, 2, width_ratios=[1, 0.4])
		axes_right = subfig_right.subplots(ndim, 1) 

		#Define colors for filtered walkers vs unfiltered
		c_fil  = '#E4003A' #red
		c_raw  = '#003285' #blue
		
		# Plot the trace for each parameter
		for i in range(ndim):

			#Define plots and setup
			ax1 = axes_left[i,0] # Step vs Walkers plot
			ax2 = axes_left[i,1] # Walkers histogram
			ax3 =  axes_right[i] # Walkers histrogram zoom in

			ax1.set_xlim((0,nsteps))
			ax1.set_ylabel(self.mcmc_variables[i])
			
			if i != ndim-1:
				ax1.set_xticklabels([])
				ax1.set_xticks([])
			else:
				ax1.set_xlabel('Step')

			ax2.set_xticklabels([])
			ax2.set_yticklabels([])
			ax2.set_yticks([])
			ax2.set_xticks([])
			
			ax3.set_xticks([])
			ax3.set_yticks([])
			ax3.set_yticklabels([])

			#Plot data and limits for the walker selction
			ax1.plot(chain[:, :, i], alpha=0.5)
			
			#Calculate histogram for original and filtered walker
			ith_walker = flat_chain[:,i]
			ith_filt_walker = flat_filt_chain[:,i]

			hist, bins   = np.histogram(ith_walker, bins='fd', density=True)
			fhist, fbins = np.histogram(ith_filt_walker, bins='fd',density=True)
			
			m = mu_list[i]
			s = sigma_list[i]

			l_lim = m-sigma*s
			h_lim = m+sigma*s

			#Plot histograms
			hist_params = {
				'density': True,
				'orientation': 'horizontal'
			}

			for t,a in zip(['step', 'stepfilled'],[1,0.2]):
				ax2.hist(ith_walker, color=c_raw, bins=bins, **hist_params, histtype=t, alpha=a)
				ax2.hist(ith_filt_walker, color=c_fil, bins=fbins, **hist_params, histtype=t, alpha=a)

				ax3.hist(ith_walker, color=c_raw, bins=bins, **hist_params, histtype=t, alpha=a)
				ax3.hist(ith_filt_walker, color=c_fil, bins=fbins, **hist_params, histtype=t, alpha=a)

			#Plot gaussian
			x = np.linspace(l_lim,h_lim,100)
			ax3.plot(gauss(x,np.max(fhist),m,s), x, ls='-.',c='k')

			ax1.vlines(x=step_cut, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], ls='--',lw=0.9, zorder=0, color='k', alpha=0.5)
			ax1.hlines(y=l_lim, xmin=step_cut, xmax=nsteps, ls='-.', zorder=0, lw=0.8, color='k')
			ax1.hlines(y=h_lim, xmin=step_cut, xmax=nsteps, ls='-.', zorder=0, lw=0.8, color='k')
			
			rect = patches.Rectangle(
				xy = (step_cut, l_lim), 
				width=(nsteps-step_cut), 
				height=(h_lim-l_lim), 
				linewidth=0, 
				edgecolor=None, 
				facecolor=c_fil,
				alpha=0.2)
			
			ax1.add_patch(rect)
			
			# Mark zoomed-in range in column 2 and set limits
			ax2.axhline(l_lim, color='black', linestyle='--', linewidth=1)
			ax2.axhline(h_lim, color='black', linestyle='--', linewidth=1)
			
			ax2.set_ylim(ax1.get_ylim()[0],ax1.get_ylim()[1])
			ax3.set_ylim(l_lim,h_lim)

			#Add patches connecting plots	
			for bound in np.sort([l_lim, h_lim]):
				line = patches.ConnectionPatch(
					xyA=(ax2.get_xlim()[1], bound), 
					xyB=(ax3.get_xlim()[0], bound), 
					coordsA="data", coordsB="data",
					axesA=ax2, axesB=ax3, color="black", linestyle="dashed"
					)
				
				fig.add_artist(line)

		subfig_left.subplots_adjust(hspace=0,wspace=0)
		subfig_right.subplots_adjust(hspace=0.02,wspace=0)
		
		plt.tight_layout()

		#Estimate best fit parameters
		fit_parameters = self.params.copy()

		best_fit = np.median(flat_filt_chain, axis=0) 
		uncertainties = np.std(flat_filt_chain, axis=0)

		for idx, elem in enumerate(self.mcmc_variables):
			fit_parameters[elem].value  = best_fit[idx]
			fit_parameters[elem].stderr = uncertainties[idx]

		#Store variables
		self.chain_analysed_by_user = True
		self.mcmc_chains = chain
		self.clean_chain = flat_filt_chain
		self.step_cut = step_cut
		self.sigma = sigma
		self.thin = thin
		self.fit_parameters = fit_parameters

		return fig, (axes_left, axes_right)

	def fit_to_file(
			self,
			results_dir='mcmc_results',
			appendix='fit_result',
			skip_analysis=False,
			overwrite=False,
			):
		
		if skip_analysis:
			pass
		elif not self.chain_analysed_by_user:
			raise ValueError(f' Please analyse the walkers by calling `GraviFit.inspect_walkers`')

		if not os.path.exists(results_dir):
			os.makedirs(results_dir)

		output_name = os.path.join(results_dir, self.idata.filename[:-5] + '_' + appendix + '.h5')
	
		if os.path.exists(output_name) and not overwrite:
			raise ValueError(f"File '{output_name}' already exists. Choose a different appendix, remove the existing file or use the keyword `overwrite`.")

		with h5py.File(output_name, "w") as f:

			#
			# Data group
			#

			idata = self.idata

			grp = f.create_group("data")

			#Header information
			grp.attrs['filename'] 		= idata.filename
			grp.attrs['date_obs'] 		= idata.date_obs
			grp.attrs['object'  ] 		= idata.object
			grp.attrs['ra'  	] 		= idata.ra
			grp.attrs['dec'  	] 		= idata.dec
			grp.attrs['sobj'  	] 		= idata.sobj

			grp.attrs['sobj_x'] = idata.sobj_x
			grp.attrs['sobj_y'] = idata.sobj_y
			
			grp.attrs['sobj_offx'] = idata.sobj_offx
			grp.attrs['sobj_offy'] = idata.sobj_offy

			#Metrology quantities
			grp = f.create_group("data/metrology")
			
			for key in ['GV1', 'GV2', 'GV3', 'GV4']:
				grp.create_dataset(key+'_offx'		, data=idata.sobj_metrology_correction_x[key])
				grp.create_dataset(key+'_offy'		, data=idata.sobj_metrology_correction_y[key])
				grp.create_dataset(key+'_northangle', data=idata.north_angle[key])

			#Baselines
			grp = f.create_group("data/array")
			
			grp.create_dataset('Bu' 	, data=idata.Bu)
			grp.create_dataset('Bv' 	, data=idata.Bv)
			grp.create_dataset('telpos' , data=idata.tel_pos)
			
			#Note: To save the strings to hdf one needs to convert them to byte strings
			grp.create_dataset('bl_telescopes' , data= [arr.astype('S') for arr in idata.bl_telescopes])
			grp.create_dataset('t3_telescopes' , data= [arr.astype('S') for arr in idata.t3_telescopes])

			grp.create_dataset('bl_labels' , data= idata.bl_labels.astype('S'))
			grp.create_dataset('t3_labels' , data= idata.t3_labels.astype('S'))
	
			grp = f.create_group("data/visibility/wave")
			grp.create_dataset('polarization' 	, data= idata.pol)
			grp.create_dataset('wave' 			, data= idata.wave)
			grp.create_dataset('band' 			, data= idata.band)
			
			grp = f.create_group("data/visibility/vis")
			grp.create_dataset('visamp' 	, data= idata.visamp)
			grp.create_dataset('visamp_err' , data= idata.visamp_err)			
			grp.create_dataset('visphi' 	, data= idata.visphi)
			grp.create_dataset('visphi_err' , data= idata.visphi_err)
			grp.create_dataset('vis_flag' 	, data= idata.vis_flag)

			grp = f.create_group("data/visibility/vis2")
			grp.create_dataset('vis2' 		, data= idata.vis2)
			grp.create_dataset('vis2_err' 	, data= idata.vis2_err)			
			grp.create_dataset('vis2_flag' 	, data= idata.vis2_flag)

			grp = f.create_group("data/visibility/t3")
			grp.create_dataset('t3amp' 		, data= idata.t3amp)
			grp.create_dataset('t3amp_err' 	, data= idata.t3amp_err)			
			grp.create_dataset('t3phi' 		, data= idata.t3phi)
			grp.create_dataset('t3phi_err' 	, data= idata.t3phi_err)			
			grp.create_dataset('t3_flag' 	, data= idata.t3flag)

			grp = f.create_group("data/visibility/sf")
			grp.create_dataset('spatial_frequency', data= idata.spatial_frequency_as)
			grp.create_dataset('spatial_frequency_t3', data= idata.spatial_frequency_as_T3)

			#
			# Fit group
			#

			grp = f.create_group("fit")
		
			grp.attrs['fit_performed'] = self.fit_performed
			grp.attrs['parameters_setup'] = self.parameters_setup

			if self.fit_performed:
				grp.attrs['Date of fit'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

			grp = f.create_group("fit/parameters")

			if self.parameters_setup:
				
				grp.attrs['field_type'] = self.field_type
				grp.attrs['nsources'] = self.nsources
				grp.attrs['use_phasemaps'] = self.use_phasemaps 

				if self.use_phasemaps:
					grp.attrs['phasemap_year'] = self.phasemap_year 
					grp.attrs['phasemap_smoothing_kernel'] = self.phasemap_smoothing_kernel
		
			grp = f.create_group("fit/parameters/initial")

			if self.parameters_setup:
			
				for name, par in self.params.items():
					grp.create_dataset(name, data=np.array([
					par.value, par.min, par.max, int(par.vary), par.stderr
					]).astype(np.float64))

			grp = f.create_group("fit/parameters/result")
			
			if self.chain_analysed_by_user:
			
				for name, par in self.fit_parameters.items():
					grp.create_dataset(name, data=np.array([
					par.value, par.min, par.max, int(par.vary), par.stderr,
					]).astype(np.float64))
			
			grp = f.create_group("fit/emcee")
			
			nsteps, nwalkers, ndim = self.mcmc_chains.shape

			grp.attrs['nsteps'] 	= nsteps
			grp.attrs['nwalkers'] 	= nwalkers
			grp.attrs['ndim'] 		= ndim
			grp.attrs['thin'] 		= self.thin

			grp = f.create_group("fit/emcee/full_chain")

			for idx, param in enumerate(self.mcmc_variables):
				grp.create_dataset(param, data=self.mcmc_chains[:,:,idx].astype(np.float64))

			grp = f.create_group("fit/emcee/clean_chain")

			grp.attrs['step_cut'] 	= self.step_cut
			grp.attrs['sigma_cut'] 	= self.sigma

			for idx, param in enumerate(self.mcmc_variables):
				grp.create_dataset(param, data=self.clean_chain[:,idx].astype(np.float64))

		return
	
	#========================================
	# Visualization tools and fit inspection
	#========================================
	
	def plot_interferometric_data(self):
		return plot_interferometric_data(self.idata)

	def fit_report_template(
			self, 
			wavelength=2.2, 
			fiber_fov=70,
			*,
			visamp_limits=(0.0, 1.1),
			visamp_residuals_limits = (-0.2, 0.2),
			visphi_limits=(-250, 250),
			visphi_residuals_limits=(-50, 50),
			vis2_limits=(0.0,1.1),
			vis2_residuals_limits=(-0.2,0.2),
			closure_limits=(-250, 250),
			closure_residuals_limits=(-50, 50)
			):

		A4_size = (11.69,8.27)
		
		# Create the figure
		fig = plt.figure(figsize=A4_size,layout='constrained')
		fig.suptitle(f'{self.idata.filename}', y = 0.99)
				
		#Create top and bottom row
		_ , top_row, bottom_row = fig.subfigures(3, 1, hspace=0.1,height_ratios=[0.02, 1,1.5])

		#Top row contains phasemaps and field of view with model (and possibly the dirty beam)
		phasemaps, field_of_view, baselines = top_row.subfigures(1, 3, width_ratios=[0.4, 0.22, 0.15],wspace=0.05)

		pm_axes = phasemaps.subplots(2, 4, width_ratios=[1,1,1,1.08])
		fov_ax  = field_of_view.subplots(1, 1)
		baselines_ax  = baselines.subplots(2, 1)

		#Bottom row contains data, model and residuals
		data_axes = bottom_row.subplots(2,4, height_ratios=[1,0.4], gridspec_kw = {'wspace':0, 'hspace':0})

		#---------------------
		# Setup phasemap plot
		#---------------------

		for idx,beam in enumerate(['GV1','GV2','GV3','GV4']):

			ax1, ax2 = pm_axes[:,idx]

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

			if idx==3:
				divider1 = make_axes_locatable(ax1)
				cax1 = divider1.append_axes("right", size="5%", pad="3%")
			#	cbar1 = plt.colorbar(intensity_plot, cax=cax1)

				divider2 = make_axes_locatable(ax2)
				cax2 = divider2.append_axes("right", size="5%", pad="3%")
			#	cbar2 = plt.colorbar(phase_plot, cax=cax2)


			circ = plt.Circle((0,0), radius=fiber_fov, facecolor="None", edgecolor='black', linewidth=0.8)
			ax1.add_artist(circ)

			circ = plt.Circle((0,0), radius=fiber_fov, facecolor="None", edgecolor='black', linewidth=0.8)
			ax2.add_artist(circ)

			#Add northangle orientation
			angle = self.idata.north_angle[beam]
			n0 = np.array([np.sin(angle),np.cos(angle)])
			
			p1 = (fiber_fov-10)*n0 
			p2 = (fiber_fov+10)*n0 
		
			ax1.plot([p1[0],p2[0]],[p1[1],p2[1]], c='k', lw=0.8)
			ax2.plot([p1[0],p2[0]],[p1[1],p2[1]], c='k', lw=0.8)

			ax1.set_aspect(1) 
			ax2.set_aspect(1) 

		#---------------------
		# Setup FOV plot
		#---------------------

		ax = fov_ax
		scale = 1.05*fiber_fov
		ax.set_xlim( scale,-scale)
		ax.set_ylim(-scale, scale)

		ax.set_xlabel('ra [mas]')
		ax.set_ylabel('dec [mas]')

		circ = plt.Circle((0,0), radius=fiber_fov, facecolor="None", edgecolor='black', linewidth=0.8)
		ax.add_artist(circ)

		#Add north east angle
		angle = 0.0
		n0 = np.array([np.sin(angle),np.cos(angle)])		
		p1 = (fiber_fov-10)*n0 
		p2 = (fiber_fov+10)*n0 
		ax.plot([p1[0],p2[0]],[p1[1],p2[1]], c='k', lw=0.8)
		ax.text(p1[0]-2,p1[1],'N', rotation=angle)
		#Add north east angle
		angle = np.pi/2
		n0 = np.array([np.sin(angle),np.cos(angle)])		
		p1 = (fiber_fov-10)*n0 
		p2 = (fiber_fov+10)*n0 
		ax.plot([p1[0],p2[0]],[p1[1],p2[1]], c='k', lw=0.8)
		ax.text(p1[0]+5,p1[1]+2,'E', rotation=angle)

		ax.set_aspect(1)

		#---------------------
		# Setup baselines plot
		#---------------------

		for ax in baselines_ax:
			ax.set_aspect(1)

		#---------------------
		# Setup bottom plot
		#---------------------

		#Visibility amplitude
		lim = (70,340)
		ax = data_axes[:,0]
		ax[0].set_title('Visibility Amplitude')
		ax[0].set_xticks([])
		
		ax[0].set_xlim(lim)
		ax[1].set_xlim(lim)
		
		ax[0].set_ylim(visamp_limits)
		ax[1].set_ylim(visamp_residuals_limits)
		ax[0].set_ylabel('Visibility Amplitude')
		ax[1].set_xlabel('spatial frequency (1/arcsec)')
		ax[1].set_ylabel('Residuals')
		ax[0].axhline(1, ls='--', lw=0.5, c='black')
		ax[1].axhline(0, ls='--', lw=0.5, c='black')

		#Visibility Phase plot
		ax = data_axes[:,1]
		ax[0].set_title('Visibility Phase')
		ax[0].set_xlim(lim)
		ax[1].set_xlim(lim)
		ax[0].set_xticks([])
		ax[0].set_ylim(visphi_limits)
		ax[1].set_ylim(visphi_residuals_limits)
		ax[0].set_ylabel('Visibility Phase')
		ax[1].set_xlabel('spatial frequency (1/arcsec)')
		ax[1].set_ylabel('Residuals')
		ax[0].axhline(0, ls='--', lw=0.5, c='black')
		ax[1].axhline(0, ls='--', lw=0.5, c='black')


		#Visibility Squared plot
		ax = data_axes[:,2]
		ax[0].set_title('Visibility Squared')
		ax[0].set_xlim(lim)
		ax[1].set_xlim(lim)
		ax[0].set_xticks([])
		ax[0].set_ylim(vis2_limits)
		ax[1].set_ylim(vis2_residuals_limits)
		ax[0].set_ylabel('Visibility Squared')
		ax[1].set_xlabel('spatial frequency (1/arcsec)')
		ax[1].set_ylabel('Residuals')
		ax[0].axhline(1, ls='--', lw=0.5, c='black')
		ax[1].axhline(0, ls='--', lw=0.5, c='black')

		lim2 = (150,350)
		#Closure phases
		ax = data_axes[:,3]
		ax[0].set_title('Closure Phases')
		ax[0].set_xlim(lim2)
		ax[1].set_xlim(lim2)
		ax[0].set_xticks([])
		ax[0].set_ylim(closure_limits)
		ax[1].set_ylim(closure_residuals_limits)
		ax[0].set_ylabel('Closure Phases')
		ax[1].set_xlabel('spatial frequency (1/arcsec)')
		ax[1].set_ylabel('Residuals')
		ax[0].axhline(0, ls='--', lw=0.5, c='black')
		ax[1].axhline(0, ls='--', lw=0.5, c='black')

		return fig, (pm_axes, cax1, cax2), fov_ax, baselines_ax, data_axes

	def fit_report(
			self, 
			params,
			wavelength=2.2, 
			fiber_fov=70, 
			*,
			visamp_limits=(0.0, 1.1),
			visamp_residuals_limits = (-0.2, 0.2),
			visphi_limits=(-250, 250),
			visphi_residuals_limits=(-50, 50),
			vis2_limits=(0.0,1.1),
			vis2_residuals_limits=(-0.2,0.2),
			closure_limits=(-250, 250),
			closure_residuals_limits=(-50, 50)
			):

		#Load keyword arguments for plot limits
		plot_limits = {
		'visamp_limits'			 	: visamp_limits			,	
		'visamp_residuals_limits'  	: visamp_residuals_limits  ,
		'visphi_limits'			 	: visphi_limits			,
		'visphi_residuals_limits'	: visphi_residuals_limits	,	
		'vis2_limits'			 	: vis2_limits			 	,
		'vis2_residuals_limits'	 	: vis2_residuals_limits	,
		'closure_limits'			: closure_limits			,
		'closure_residuals_limits'	: closure_residuals_limits	,
		}

		fig, (pm_axes, cax1, cax2), fov_ax, baselines_ax, data_axes = self.fit_report_template(wavelength=wavelength, fiber_fov=fiber_fov, **plot_limits)

		#Get sources
		sources, _ = GraviFit.get_sources_and_background(params.valuesdict(), self.field_type, self.nsources )

		#
		# Plot phasemaps
		#

		pltargsP = {'cmap': 'twilight_shifted', 'levels': np.linspace(-180, 180, 19, endpoint=True)}

		if self.use_phasemaps:
			pms = self.phasemaps
			
			for idx, beam in enumerate(pms):
			
				# Get data

				# There should be a try here for when there are no loaded phasemaps
				x = pms[beam].grid[1]
				y = pms[beam].grid[2]

				xx, yy = np.meshgrid(x, y)
				zz = pms[beam]((wavelength,xx,yy))
				
				rmap = np.sqrt(xx*xx + yy*yy)
				zz[rmap > fiber_fov] = 0.0 

				# Setup axis
				ax1, ax2 = pm_axes[:,idx]
			
				#Plot
				intensity_plot = ax1.pcolormesh(xx, yy, np.abs(zz)/np.max(np.abs(zz)))
				phase_plot = ax2.contourf(xx, yy, np.angle(zz,deg=True), **pltargsP)

				circ = plt.Circle((0,0), radius=fiber_fov, facecolor="None", edgecolor='black', linewidth=0.8)
				ax1.add_artist(circ)

				circ = plt.Circle((0,0), radius=fiber_fov, facecolor="None", edgecolor='black', linewidth=0.8)
				ax2.add_artist(circ)

				if idx==3:
					cbar1 = plt.colorbar(intensity_plot, cax=cax1)
					cbar2 = plt.colorbar(phase_plot, cax=cax2)

				#Plot sources with metrology corrections
				for x, y, flux, alpha in sources:

					angle = self.idata.north_angle[beam]

					offset = np.array([
						self.idata.sobj_metrology_correction_x[beam], 
						self.idata.sobj_metrology_correction_y[beam]]
						)

					Rm = np.array([	
						[np.cos(angle), -np.sin(angle)], 
				  		[np.sin(angle),  np.cos(angle)]]
						)

					xi,yi = np.matmul(Rm, [x,y] + offset)

					ax1.scatter([xi],[yi],  edgecolors='black', s=10.2)
					ax2.scatter([xi],[yi],  edgecolors='black', s=10.2)

		#
		# Plot field of view and model positions
		#

		#Get data
		idata = self.idata#

		#Get dirty beam
		_, _, (x,C) = get_dirty_beam(idata)

		ax = fov_ax

		ax.pcolormesh(x , x, C,cmap='gist_yarg', zorder=0)

		for x, y, _, _ in sources:
			ax.scatter([x],[y],  s=8.0, edgecolors='black')

		#
		# Plot baseline configuration
		#

		ax1, ax2 = baselines_ax

		uv_coordinates = np.transpose([idata.Bu,idata.Bv])

		for idx,station in enumerate(uv_coordinates):
			ax1.scatter( station[0], station[1], c=colors_baseline[idx], s=14.5)
			ax1.scatter(-station[0],-station[1], c=colors_baseline[idx], s=14.5)
			ax1.plot([-station[0],station[0]],[-station[1],station[1]], c=colors_baseline[idx],lw=1.8,ls='-')

		for tel_coord in idata.tel_pos:
			ax2.scatter(-tel_coord[0],-tel_coord[1],c='black',zorder=10,s=10)

		for idx, triangle in enumerate([(4,3,2) ,(4,3,1),(4,2,1),(3,2,1)]):

			index = np.array(triangle) - [1,1,1]

			#Define triangles
			x = -idata.tel_pos[index,0]
			y = -idata.tel_pos[index,1]
			
			centroid_x = np.mean(x)
			centroid_y = np.mean(y)

			scale_factor = 0.8

			x_inner = centroid_x + (x - centroid_x) * scale_factor
			y_inner = centroid_y + (y - centroid_y) * scale_factor
			
			ax2.plot(np.append(x_inner, x_inner[0]), np.append(y_inner, y_inner[0]), color=colors_closure[idx], linewidth=1)
			ax2.fill(x_inner, y_inner, color=colors_closure[idx], edgecolor=colors_closure[idx], linewidth=0, alpha=0.3)

		ax1.set_aspect(1)
		ax2.set_aspect(1)
		ax1.axis('off')
		ax2.axis('off')

		#
		# Plot data and residuals
		#
		
		visibility_model = self.get_visibility_model(params, use_phasemaps=self.use_phasemaps)
		
		plot_config = {
			'alpha':    0.8,
			'ms':       3.0,
			'lw':       0.8,
			'capsize':  1.0,
			'ls':       ''    
		}
		

		ax1,ax2,ax3,ax4 	= data_axes[0,:]
		ax1r,ax2r,ax3r,ax4r = data_axes[1,:]


		for idx,key in enumerate(visibility_model):

			#Get model
			mx, my = visibility_model[key]

			#Visibility ampliude
			x   = idata.spatial_frequency_as[idx] 
			y   = idata.visamp[idx] 
			yerr= idata.visamp_err[idx]
			
			ax1.errorbar(x, y, yerr, **plot_config, marker='o', color=colors_baseline[idx % 6])

			ax1.plot(mx,np.abs(my), color=colors_baseline[idx % 6])
			ax1.scatter(mx,np.abs(my),s=2, color=colors_baseline[idx % 6])
	
			ax1r.errorbar(x, y-np.abs(my), yerr, **plot_config, marker='D', color=colors_baseline[idx % 6])

			#Visibility phase
			x   = idata.spatial_frequency_as[idx] 
			y   = idata.visphi[idx] 
			yerr= idata.visphi_err[idx]

			ax2.errorbar(x, y, yerr, **plot_config, marker='o', color=colors_baseline[idx % 6])

			ax2.plot(mx,np.angle(my,deg=True), color=colors_baseline[idx % 6])
			ax2.scatter(mx,np.angle(my,deg=True),s=2, color=colors_baseline[idx % 6])
	
			ax2r.errorbar(x, y-np.angle(my,deg=True), yerr, **plot_config, marker='D', color=colors_baseline[idx % 6])

			#Visibility squared
			x   = idata.spatial_frequency_as[idx] 
			y   = idata.vis2[idx] 
			yerr= idata.vis2_err[idx]

			ax3.errorbar(x, y, yerr, **plot_config, marker='o', color=colors_baseline[idx % 6])

			ax3.plot(mx,np.abs(my)**2, color=colors_baseline[idx % 6])
			ax3.scatter(mx,np.abs(my)**2,s=2, color=colors_baseline[idx % 6])
	
			ax3r.errorbar(x, y-np.abs(my)**2, yerr, **plot_config, marker='D', color=colors_baseline[idx % 6])

		#Closure phases
		c1 = np.angle(visibility_model['UT43'][1]) + np.angle(visibility_model['UT32'][1]) - np.angle(visibility_model['UT42'][1])
		c2 = np.angle(visibility_model['UT43'][1]) + np.angle(visibility_model['UT31'][1]) - np.angle(visibility_model['UT41'][1])
		c3 = np.angle(visibility_model['UT42'][1]) + np.angle(visibility_model['UT21'][1]) - np.angle(visibility_model['UT41'][1])
		c4 = np.angle(visibility_model['UT32'][1]) + np.angle(visibility_model['UT21'][1]) - np.angle(visibility_model['UT31'][1])

		cp = np.array([c1,c2,c3,c4])*180/np.pi

		for idx in range(len(idata.spatial_frequency_as_T3)):

			x   = idata.spatial_frequency_as_T3[idx] 
			y   = idata.t3phi[idx] 
			yerr= idata.t3phi_err[idx]

			ax4.errorbar(x, y, yerr, **plot_config, marker='o', color=colors_closure[idx])

			ax4.plot(x,cp[idx], color=colors_closure[idx])
			ax4.scatter(x,cp[idx], s=2, color=colors_closure[idx])
	
			ax4r.errorbar(x, y-cp[idx], yerr, **plot_config, marker='D', color=colors_closure[idx])

		return fig, pm_axes, fov_ax, baselines_ax, data_axes
