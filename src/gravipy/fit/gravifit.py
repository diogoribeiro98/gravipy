import numpy as np
import copy
import pandas as pd
import os

#Parent classes
from ..data import GravData
from ..phasemaps import GravPhaseMaps
from ..data import GravData_scivis

#Units
from ..physical_units import units as units

#Fitting tools
from .models import spectral_visibility
import lmfit
import emcee
import multiprocessing

#Plotting tools
import corner
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Logging tools
import logging
from ..logger.log import log_level_mapping
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from datetime import datetime

#Alias between beam and telescope
telescope_to_beam = {
	'UT1' : 'GV4',
	'UT2' : 'GV3',
	'UT3' : 'GV2',
	'UT4' : 'GV1',
}

class GraviFit(GravPhaseMaps,GravData_scivis):
	"""GRAVITY single night fit class
	"""

	def __init__(self, file, loglevel='INFO'):	
		
		#Create a logger and set log level according to user
		self.logger = logging.getLogger(type(self).__name__)
		self.logger.setLevel(log_level_mapping.get(loglevel, logging.INFO))
		
		#Super constructor
		GravPhaseMaps.__init__(self,loglevel=loglevel)
		GravData_scivis.__init__(self,file)
		
		# ---------------------------
		# Pre-define class quantities
		# ---------------------------
		
		#Status variables
		self.parameters_setup 	= False
		self.fit_performed 		= False

		#Fitting parameters
		self.params = None
		
		#Fitting helper quantities
		self.field_type = None
		self.nsource 	= None
		self.sources 	= None
		self.background = None
		
		#Phasemap helper quantities
		self.use_phasemaps = None
		self.phasemap_year = None
		self.phasemap_smoothing_kernel = None


		#
		# Parameters only accessible after a fit is performed
		#

		self.sampler 			= None
		self.fit_params 		= None
		self.result_params 		= None
		self.fitted_pol 		= None
		self.fit_weights 		= None
		self.flagged_channels	= []
	
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
				vary  = parameter_dictionary[name][3],
				min   = parameter_dictionary[name][1],
				max   = parameter_dictionary[name][2],
				)

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
			
	def get_visibility_model(self, params, pol='P1', use_phasemaps=False):
		
		#Get baseline info from datafile and create template visibility model
		idata = self.get_interferometric_data(pol)

		visibility_model = dict((str(name),None) for name in idata.bl_labels)	

		#Get sources from parameters
		sources, background = GraviFit.get_sources_and_background(params.valuesdict(), self.field_type, self.nsource )

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

				met_offx = [ self.sobj_metrology_correction_x[telescope_to_beam[telescopes[0]]],
							 self.sobj_metrology_correction_x[telescope_to_beam[telescopes[1]]]]

				met_offy = [ self.sobj_metrology_correction_y[telescope_to_beam[telescopes[0]]],
							 self.sobj_metrology_correction_y[telescope_to_beam[telescopes[1]]]]

				nangle = [	self.north_angle[telescope_to_beam[telescopes[0]]],
							self.north_angle[telescope_to_beam[telescopes[1]]]]

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
			polarization='P1', 
			flag_channels=[],
			fit_weights=[1.0, 1.0, 0.0, 0.0]):
		"""_summary_

		Args:
			nwalkers (int, optional): _description_. Defaults to 50.
			steps (int, optional): _description_. Defaults to 100.
			nthreads (int, optional): _description_. Defaults to 1.
			initial_spread (float, optional): _description_. Defaults to 0.5.
			polarization (str, optional): _description_. Defaults to 'P1'.
			flag_channels (list, optional): _description_. Defaults to [].
			fit_weights (list, optional): _description_. Defaults to [1.0, 1.0, 0.0, 0.0].

		Returns:
			_type_: _description_
		"""

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
			nsources = self.nsource
			
			parameter_value_dictionary = self.params.valuesdict()

			idata = self.get_interferometric_data(
				pol=polarization,
				flag_channels=flag_channels)

			visibility_model = dict((str(name),None) for name in idata.bl_labels)

			use_phasemaps 			= self.use_phasemaps
			phasemaps_phase 		= self.phasemaps_phase
			phasemaps_amplitude 	= self.phasemaps_amplitude
			phasemaps_normalization = self.phasemaps_normalization

			north_angle 	= self.north_angle
			metrology_offx 	= self.sobj_metrology_correction_x
			metrology_offy 	= self.sobj_metrology_correction_y
			
			weights = fit_weights

			return			

		#
		# Perform parallel fit
		#


		log_likelihood = GraviFit.log_likelihood
		#Perform fit		
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

		#Estimate fit parameters ignoring the first 20 percent of the chain
		samples = sampler.get_chain(discard=int(0.2*steps), thin=10, flat=True) 
		best_fit = np.median(samples, axis=0) 
		uncertainties = np.std(samples, axis=0)

		#Overwrite parameter values
		result_parameters = self.params.copy()

		for idx, elem in enumerate(parameters_to_fit):
			result_parameters[elem].value  = best_fit[idx]
			result_parameters[elem].stderr = uncertainties[idx]

		#Setup class variables associated with fit
		self.flagged_channels  	= flag_channels
		self.sampler 			= sampler
		self.fit_params 		= parameters_to_fit
		self.result_params 		= result_parameters
		self.fit_performed 		= True
		self.fitted_pol 		= polarization
		self.fit_weights    	= fit_weights

		return sampler, parameters_to_fit, result_parameters

	#========================================
	# mcmc analysis tools
	#=========================================

	def get_flat_chain(self, step_cut, thin=1, sigma=5, plot=True, plot_save_name=None):
		
		if not self.fit_performed:
			raise ValueError('No fit has been performed.')

		# Part 1: Get the full chain
		full_chain = self.sampler.get_chain(thin=thin)
		full_chain_flat = self.sampler.get_chain(thin=thin, flat=True)
		n_dim = full_chain.shape[-1]
		n_steps = full_chain.shape[0]

		# Part 2: Get filtered chain according to sigma value
		discard_chain_flat = self.sampler.get_chain(discard=step_cut, thin=thin, flat=True) 

		# Calculate the median and standard deviation for each parameter
		medians = np.median(discard_chain_flat, axis=0)
		std_devs = np.std(discard_chain_flat, axis=0)

		# Define the sigma bounds
		lower_bound = medians - sigma * std_devs
		upper_bound = medians + sigma * std_devs

		# Mask out walkers outside the sigma interval for each parameter
		mask = np.all((discard_chain_flat >= lower_bound) & (discard_chain_flat <= upper_bound), axis=1)
		filtered_chain_flat = discard_chain_flat[mask]

		if plot:

			# Compute the 16th and 84th percentiles for the filtered chain
			q_low, q_high = np.percentile(filtered_chain_flat, [16, 84], axis=0)

			fig = plt.figure(figsize=(8, 2*n_dim), dpi=100)

			subfig_left, subfig_right = fig.subfigures(1, 2, wspace=0.1, width_ratios=[3, 1])  
			axes_left  = subfig_left.subplots(n_dim, 2, width_ratios=[1, 0.4])
			axes_right = subfig_right.subplots(n_dim, 1) 

			#Define colors for filtered walkers vs unfiltered
			c1 = '#E4003A'
			c2 = '#003285'

			# Plot the trace for each parameter
			for i in range(n_dim):

				#Define plots and setup
				ax1 = axes_left[i,0] # Step vs Walkers plot
				ax2 = axes_left[i,1] # Walkers histogram
				ax3 =  axes_right[i] # Walkers histrogram zoom in

				ax1.set_xlim((0,n_steps))
				ax1.set_ylabel(self.fit_params[i])

				if i != n_dim-1:
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
				ax1.plot(full_chain[:, :, i], alpha=0.5)
				
				ax1.vlines(x=step_cut, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], ls='-.',lw=0.8, zorder=0, color='k', alpha=0.5)
				ax1.vlines(x=step_cut, ymin=lower_bound[i], ymax=upper_bound[i], ls='-.',lw=0.8, zorder=0, color='k')
				ax1.hlines(y=lower_bound[i], xmin=step_cut, xmax=n_steps, ls='-.', zorder=0, lw=0.8, color='k')
				ax1.hlines(y=upper_bound[i], xmin=step_cut, xmax=n_steps, ls='-.', zorder=0, lw=0.8, color='k')
				
				rect = patches.Rectangle(
					xy = (step_cut, lower_bound[i]), 
					width=(n_steps-step_cut), 
					height=(upper_bound[i]-lower_bound[i]), 
					linewidth=0, 
					edgecolor=None, 
					facecolor=c1,
					alpha=0.2)

				ax1.add_patch(rect)

				#Plot histogram data
				chain_filt = filtered_chain_flat[:, i]
				chain_full = full_chain_flat[:, i]

				_, bins_filt = np.histogram(chain_filt, bins='fd')
				_, bins_full = np.histogram(chain_full, bins='fd')
				
				hist_params = {
					'density': False,
					'orientation': 'horizontal'
				}

				for t,a in zip(['step', 'stepfilled'],[1,0.2]):
					ax2.hist(chain_full, color=c2, bins=bins_full, **hist_params, histtype=t, alpha=a)
					ax2.hist(chain_filt, color=c1, bins=bins_filt, **hist_params, histtype=t, alpha=a)
				
				# Mark zoomed-in range in column 2 and set limits
				ax2.axhline(lower_bound[i], color='black', linestyle='--', linewidth=1)
				ax2.axhline(upper_bound[i], color='black', linestyle='--', linewidth=1)
				
				ax2.set_ylim(ax1.get_ylim()[0],ax1.get_ylim()[1])

				#Plot zoomed in histograms
				for t,a in zip(['step', 'stepfilled'],[1,0.2]):
					ax3.hist(chain_full, color=c2, bins=bins_full, **hist_params, histtype=t, alpha=a)
					ax3.hist(chain_filt, color=c1, bins=bins_filt, **hist_params, histtype=t, alpha=a)
				
				ax3.set_ylim(lower_bound[i], upper_bound[i])

				#Add patches connecting plots	
				for bound in [lower_bound[i], upper_bound[i]]:
				
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

			if plot_save_name is not None:
				plt.savefig(plot_save_name, bbox_inches="tight")
			
			plt.show()

		return filtered_chain_flat

	def corner_plot(self, step_cut, thin=1, sigma=5):
		
		if not self.fit_performed:
			raise ValueError('No fit has been performed.')

		#Get filtered samples
		filtered_chain = self.get_flat_chain(
			step_cut=step_cut, 
			thin=thin, sigma=sigma, 
			plot=False)

		fig = plt.figure(figsize=(15, 15))
		
		corner.corner(filtered_chain, 
			  fig=fig, 
			  labels=self.fit_params, 
			  quantiles=[0.16, 0.5, 0.84], 
			  show_titles=True, 
			  smooth=True, 
			  title_fmt=None, 
			  bins=int(1+np.log2(filtered_chain.shape[0]*filtered_chain.shape[1])),
			  plot_datapoints=True
			  )
		
		return fig

	def get_fit_metadata(self,step_cut, thin, sigma):
		
		metadata = [
		f"# MCMC Chain Results",
		f"# Data",
		f"# Filename: {self.filename}",
		f"# Flagged Channels: {self.flagged_channels}",
		f"# Polarization: {self.fitted_pol}",
		f"#",
		f"# Fitting info",
		f"# Date of fit: {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}",
		f"# Fit Weights: {self.fit_weights}",
		f"# Walkers: {self.sampler.nwalkers}",
		f"# Iterations: {self.sampler.iteration}",
		f"# Saved samples info",
		f"# Step Cut: {step_cut}, Thin: {thin}, Sigma: {sigma}",
		f"# Parameters: {', '.join(self.fit_params)}",
		f"# --------------------------",
		]

		return metadata

	def save_mcmc_chain_csv(self, step_cut, thin=1, sigma=5, appendix=None):
		"""Save mcmc chain to csv file

		Args:
			step_cut (_type_): _description_
			thin (int, optional): _description_. Defaults to 1.
			sigma (int, optional): _description_. Defaults to 5.
			appendix (_type_, optional): _description_. Defaults to None.

		Raises:
			ValueError: _description_
			ValueError: _description_
		"""
		
		if appendix==None:
			raise ValueError('Please provide an appendix to save the mcmc chain')
		
		results_dir = 'mcmc_results'
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)

		output_name = os.path.join(results_dir, self.filename[:-5] + '_' + appendix + '.csv')

		if os.path.exists(output_name):
			raise ValueError(f"File '{output_name}' already exists. Choose a different appendix or remove the existing file.")

		#Get filtered samples
		filtered_chain = self.get_flat_chain(
			step_cut=step_cut, 
			thin=thin, sigma=sigma, 
			plot=False)

		# Convert to DataFrame with column names
		df = pd.DataFrame(filtered_chain, columns=self.fit_params)

		#Get fit metadata
		metadata = self.get_fit_metadata(step_cut=step_cut,thin=thin,sigma=sigma)
				
		with open(output_name, "w") as f:
			for line in metadata:
				f.write(line + "\n") 
		
		df.to_csv(output_name, index=False,  mode='a')

		return
	
	def load_mcmc_csv(self, filename):
		return pd.read_csv(filename, comment='#')

	#========================================
	# Visualization tools and fit inspection
	#========================================

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
		fig.suptitle(f'{self.filename}', y = 0.99)
				
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
			angle = self.north_angle[beam]
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
			pol,
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
		sources, _ = GraviFit.get_sources_and_background(params.valuesdict(), self.field_type, self.nsource )

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

					angle = self.north_angle[beam]

					offset = np.array([
						self.sobj_metrology_correction_x[beam], 
						self.sobj_metrology_correction_y[beam]]
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
		idata = self.get_interferometric_data(
			pol=pol,
			flag_channels=self.flagged_channels)

		ax = fov_ax
		ax.set_facecolor('#8f8f8f')

		for x, y, flux, alpha in sources:
			ax.scatter([x],[y],  edgecolors='black')

		#
		# Plot baseline configuration
		#

		ax1, ax2 = baselines_ax

		uv_coordinates = np.transpose([idata.Bu,idata.Bv])

		for idx,station in enumerate(uv_coordinates):
			ax1.scatter( station[0], station[1], c=self.colors_baseline[idx], s=14.5)
			ax1.scatter(-station[0],-station[1], c=self.colors_baseline[idx], s=14.5)
			ax1.plot([-station[0],station[0]],[-station[1],station[1]], c=self.colors_baseline[idx],lw=1.8,ls='-')

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
			
			ax2.plot(np.append(x_inner, x_inner[0]), np.append(y_inner, y_inner[0]), color=self.colors_closure[idx], linewidth=1)
			ax2.fill(x_inner, y_inner, color=self.colors_closure[idx], edgecolor=self.colors_closure[idx], linewidth=0, alpha=0.3)

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
			
			ax1.errorbar(x, y, yerr, **plot_config, marker='o', color=self.colors_baseline[idx % 6])

			ax1.plot(mx,np.abs(my), color=self.colors_baseline[idx % 6])
			ax1.scatter(mx,np.abs(my),s=2, color=self.colors_baseline[idx % 6])
	
			ax1r.errorbar(x, y-np.abs(my), yerr, **plot_config, marker='D', color=self.colors_baseline[idx % 6])

			#Visibility phase
			x   = idata.spatial_frequency_as[idx] 
			y   = idata.visphi[idx] 
			yerr= idata.visphi_err[idx]

			ax2.errorbar(x, y, yerr, **plot_config, marker='o', color=self.colors_baseline[idx % 6])

			ax2.plot(mx,np.angle(my,deg=True), color=self.colors_baseline[idx % 6])
			ax2.scatter(mx,np.angle(my,deg=True),s=2, color=self.colors_baseline[idx % 6])
	
			ax2r.errorbar(x, y-np.angle(my,deg=True), yerr, **plot_config, marker='D', color=self.colors_baseline[idx % 6])

			#Visibility squared
			x   = idata.spatial_frequency_as[idx] 
			y   = idata.vis2[idx] 
			yerr= idata.vis2_err[idx]

			ax3.errorbar(x, y, yerr, **plot_config, marker='o', color=self.colors_baseline[idx % 6])

			ax3.plot(mx,np.abs(my)**2, color=self.colors_baseline[idx % 6])
			ax3.scatter(mx,np.abs(my)**2,s=2, color=self.colors_baseline[idx % 6])
	
			ax3r.errorbar(x, y-np.abs(my)**2, yerr, **plot_config, marker='D', color=self.colors_baseline[idx % 6])

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

			ax4.errorbar(x, y, yerr, **plot_config, marker='o', color=self.colors_closure[idx])

			ax4.plot(x,cp[idx], color=self.colors_closure[idx])
			ax4.scatter(x,cp[idx], s=2, color=self.colors_closure[idx])
	
			ax4r.errorbar(x, y-cp[idx], yerr, **plot_config, marker='D', color=self.colors_closure[idx])

		return fig, pm_axes, fov_ax, baselines_ax, data_axes
