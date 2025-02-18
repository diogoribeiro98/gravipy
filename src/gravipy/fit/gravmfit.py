import numpy as np

#Logging tools
import logging
from ..logger.log import log_level_mapping

#Parent classes
from ..data import GravData
from ..phasemaps import GravPhaseMaps

#Units
from ..physical_units import units as units

#Fitting tools
from .models import spectral_visibility

#Alias between beam and telescope
telescope_to_beam = {
	'UT1' : 'GV4',
	'UT2' : 'GV3',
	'UT3' : 'GV2',
	'UT4' : 'GV1',
}

class GravMfit(GravData, GravPhaseMaps):
	"""GRAVITY single night fit class
	"""

	def __init__(self, data, loglevel='INFO'):	
		
		#Create a logger and set log level according to user
		self.logger = logging.getLogger(type(self).__name__)
		self.logger.setLevel(log_level_mapping.get(loglevel, logging.INFO))
		
		#Super constructor
		GravData.__init__(self,data,loglevel=loglevel)
		GravPhaseMaps.__init__(self,loglevel=loglevel)
		
		# ---------------------------
		# Pre-define class quantities
		# ---------------------------
		
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

		#Template structure for visibility model
		self.visibility_model = dict((str(name),None) for name in self.baseline_labels)

		#Baseline index maps 
		self.baseline_index_map = {label : idx for idx, label in enumerate(self.baseline_labels)}

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
			self.nsource = nsource
			self.sources = np.zeros((nsource, 4)) 	# [ra,dec,flux,alpha]
			self.background = np.zeros(2)			# [flux,alpha]
			
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
					star_fit_parameters[f'source_{idx}_dmag'] = [ 0.0, np.log10(0.001),  np.log10(100.), False ]
				else:
					star_fit_parameters[f'source_{idx}_dmag'] = [ -2.5*np.log10(fr_list[idx]/fr_list[0]),  -4,  4, fit_star_fr ]

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
				self.nsource = 0
				self.sources = np.zeros((1, 4)) # [ra,dec,flux,alpha]
				self.background = np.zeros(2)	# [flux,alpha]

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
					'sgra_dmag'  : [0.0			, np.log10(0.001), np.log10(100.), False        ],
					'sgra_alpha' : [sgr_alpha	, -10.0	 		 , 10.0			 , fit_sgr_alpha],
				}

				#Fitting parameters for background
				# number of parameters = 2
				background_fit_parameters = {
					'background_flux' : [background_fr   ,  0.0, 20.0, fit_background_fr   ],
					'background_alpha': [background_alpha, -10.0, 10.0, fit_background_alpha]
				}	

				all_fitting_parameters = {}
				all_fitting_parameters.update(sgra_fit_parameters) 
				all_fitting_parameters.update(background_fit_parameters)

				self.params = self.assemble_parameter_class(all_fitting_parameters)

			else:

				#Setup class and create list of sources
				self.field_type = 'sgra' 
				self.nsource = nsource
				self.sources = np.zeros((nsource + 1, 4))  	# [ra,dec,flux,alpha]
				self.background = np.zeros(2)				# [flux,alpha]

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
						star_fit_parameters[f'source_{idx}_dmag'] = [ 0.0, 0.0,  10. , False ]
					else:
						star_fit_parameters[f'source_{idx}_dmag'] = [ -2.5*np.log10(fr_list[idx]/fr_list[0]), -4 ,  4 , fit_star_fr ]

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
					'sgra_dmag'  : [-2.5*np.log10(sgr_fr/fr_list[0])	,   -4			 , 4		   	 , fit_sgr_fr   ],
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
			normalization_maps = None
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
			xlist = np.zeros_like(l_list)
			ylist = np.zeros_like(l_list)

			for src in sources:

				#Get position, flux and spectral index for each source
				x, y, flux, alpha = src
			
				#Calculate optical path difference from position on sky and visibility
				opd.fill((u*x + v*y)*units.mas_to_rad)
				
				xlist.fill(x)
				ylist.fill(y)
				
				#Correct with phasemaps
				arg = (l_list,xlist,ylist)
				opd -= ( phi_i(arg) - phi_j(arg))*l_list/360. 
				Lij  =  Ai(arg)*Aj(arg)

				#Calculate visibility
				visibility += Lij*flux*spectral_visibility(opd, alpha, l_list, dl_list, reference_l0)
				
				# Normalization terms
				sv0 = spectral_visibility(0, alpha, l_list, dl_list, reference_l0)
				
				normalization_i += Li(arg) * flux * sv0
				normalization_j += Lj(arg) * flux * sv0
				
		 	#Add background to normalization
			flux, alpha = background
			normalization_i += flux*spectral_visibility(0, alpha, l_list, dl_list,reference_l0)
			normalization_j += flux*spectral_visibility(0, alpha, l_list, dl_list,reference_l0)
			
			#Calculate spatial frequencies
			sf = np.sqrt(u**2+v**2)/l_list*units.as_to_rad
			
			return sf, visibility/np.sqrt(normalization_i*normalization_j)
			
	def get_visibility_model(self, params, use_phasemaps=False):
		
		#Get sources from parameters
		sources, background = GravMfit.get_sources_and_background(params.valuesdict(), self.field_type, self.nsource )

		#Storage dictionary
		visibility_model = copy.deepcopy(self.visibility_model)

		for telescopes, label in zip(self.baseline_telescopes, self.baseline_labels):
			
			baseline_index = self.baseline_index_map[label] 

			ucoord = self.u[baseline_index]/units.micrometer
			vcoord = self.v[baseline_index]/units.micrometer

			#Load phasemap interpolating functions if using phasemaps
			if use_phasemaps:

				phase_maps = [self.phasemaps_phase[telescope_to_beam[telescopes[0]]],
							  self.phasemaps_phase[telescope_to_beam[telescopes[1]]]]
						
				amplitude_maps = [	self.phasemaps_amplitude[telescope_to_beam[telescopes[0]]],
									self.phasemaps_amplitude[telescope_to_beam[telescopes[1]]]]
						
				normalization_maps = [	self.phasemaps_normalization[telescope_to_beam[telescopes[0]]],
										self.phasemaps_normalization[telescope_to_beam[telescopes[1]]]]

				phasemap_args = {
					'phase_maps': phase_maps,
					'amplitude_maps': amplitude_maps,
					'normalization_maps': normalization_maps
				}

			else:
				phasemap_args = {}

			model = GravMfit.nsource_visibility(
			uv_coordinates= [ucoord,vcoord],
			sources=sources,
			background=background,
			l_list=self.wlSC,
			dl_list=self.dlambda,
			use_phasemaps=use_phasemaps,
			**phasemap_args
			)

			visibility_model[label] = model
				
		return visibility_model

	#========================================
	# Static methods for multithread fitting
	#=========================================
	
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
	def emcee_log_likelihood(theta, theta_names):
		"""
		Loglikelihood function for emcee sampler.
		
		The functions is made to run in parallel and needs
		access to global variables. As such it should only be
		called from within an open pool of workers with the proper
		emcee worker initializer (see GravMfit.run_emcee_fit) function

		"""
	
		# Create a parameter dictionary with all parameters + theta values
		global parameter_value_dictionary
		global uniform_prior_bounds

		theta_parameter_dictionary = dict(zip(theta_names, theta))

		for key in parameter_value_dictionary:
			parameter_value_dictionary[key] = theta_parameter_dictionary.get(key, parameter_value_dictionary[key])

		#Check if within the priors and reject samples outside
		for key, value in theta_parameter_dictionary.items():
			if key in uniform_prior_bounds:
				min_bound, max_bound = uniform_prior_bounds[key]
				if not (min_bound <= value <= max_bound):
					return -np.inf  

		#Fetch sources and background from the parameter dictionary
		global field_type
		global nsources

		sources, background = GravMfit.get_sources_and_background(parameter_value_dictionary,field_type=field_type,nsource=nsources)

		#Calculate visibilities
		global baseline_telescopes
		global baseline_labels
		global baseline_index_map
		
		global wavelength_vector, dlambda_vector
		global phasemaps_phase, phasemaps_amplitude, phasemaps_normalization
		global use_phasemaps
	
		global visibility_model
	
		for telescopes, label in zip(baseline_telescopes, baseline_labels):
			
			baseline_index = baseline_index_map[label] 

			ucoord = u[baseline_index]/units.micrometer
			vcoord = v[baseline_index]/units.micrometer

			#Load phasemap interpolating functions if using phasemaps
			if use_phasemaps:

				phase_maps = [phasemaps_phase[telescope_to_beam[telescopes[0]]],
							  phasemaps_phase[telescope_to_beam[telescopes[1]]]]
						
				amplitude_maps = [	phasemaps_amplitude[telescope_to_beam[telescopes[0]]],
									phasemaps_amplitude[telescope_to_beam[telescopes[1]]]]
						
				normalization_maps = [	phasemaps_normalization[telescope_to_beam[telescopes[0]]],
										phasemaps_normalization[telescope_to_beam[telescopes[1]]]]
				
				phasemap_args = {
					'phase_maps': phase_maps,
					'amplitude_maps': amplitude_maps,
					'normalization_maps': normalization_maps
				}

			else:
				phasemap_args = {}

			model = GravMfit.nsource_visibility(
			uv_coordinates= [ucoord,vcoord],
			sources=sources,
			background=background,
			l_list=wavelength_vector,
			dl_list=dlambda_vector,
			use_phasemaps=use_phasemaps,
			**phasemap_args		
			)
		
			visibility_model[label] = model
		
		# Compute residuals
		residual_sum = 0.0

		for idx, label in enumerate(visibility_model):
			
			# Model quantities
			visamp_model = np.abs(visibility_model[label][1])
			visphi_model = np.angle(visibility_model[label][1], deg=True)
			
			# Data pairs for amplitude, phase, and squared residuals (only P1 implemented at the moment)
			amp_data, amp_err = visamp[idx], visamp_err[idx]
			phi_data, phi_err = visphi[idx], visphi_err[idx]

			residuals_amp = ((visamp_model - amp_data)/amp_err )**2
			residual_sum += np.nansum(residuals_amp)

			residuals_phi = ((visphi_model - phi_data)/phi_err )**2
			residual_sum += np.nansum(residuals_phi)

		log_likelihood = -0.5*residual_sum

		return log_likelihood 
			
	def run_mcmc_fit(self, nwalkers=50, steps=200, nthreads=1, initial_spread = 0.5, polarization='P1'):

		#Initial spread cannot be larger than 1
		if initial_spread >= 1:
			raise ValueError('initial_spread cannot be larger than 1.')

		#Get parameter names to fit
		parameters_to_fit = [p for p in self.params if self.params[p].vary==True]
		initial_values    = [self.params[p].value for p in parameters_to_fit] 
		param_range  = [ np.abs(self.params[p].max-self.params[p].min)/2 for p in parameters_to_fit] 
		
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
		
		#Select model
		log_likelihood = GravMfit.emcee_log_likelihood

		#Define emcee worker init function to share data with each thread
		def emcee_worker_init():

			#Baseline information
			global baseline_telescopes
			global baseline_labels
			global baseline_index_map
			global u, v

			baseline_telescopes = self.baseline_telescopes
			baseline_labels = self.baseline_labels
			baseline_index_map = self.baseline_index_map
			u = self.u
			v = self.v

			#Wavelength information
			global wavelength_vector
			global dlambda_vector
			wavelength_vector = self.wlSC
			dlambda_vector = self.dlambda

			#Storage variables
			global visibility_model
			visibility_model = self.visibility_model

			#Parameter initial values and uniform bounds
			global parameter_value_dictionary
			global uniform_prior_bounds

			parameter_value_dictionary = self.params.valuesdict()
		
			uniform_prior_bounds = {
    			name: (param.min, param.max)
				for name, param in self.params.items()
				if param.min is not None and param.max is not None
				}

			#Field type and number of sources
			global field_type
			global nsources

			field_type = self.field_type
			nsources = self.nsource

			#Phasemaps
			global phasemaps_phase, phasemaps_amplitude
			global phasemaps_normalization
			global use_phasemaps

			phasemaps_phase = self.phasemaps_phase
			phasemaps_amplitude = self.phasemaps_amplitude
			phasemaps_normalization = self.phasemaps_normalization
			use_phasemaps = self.use_phasemaps

			#Data
			global visamp, visamp_err
			global visphi, visphi_err
			#global vis2  , vis2_err
			#global t3    , t3_err
			
			if polarization=='P1':
				visamp, visamp_err = self.visampSC_P1, self.visamperrSC_P1
				visphi, visphi_err = self.visphiSC_P1, self.visphierrSC_P1
			elif polarization=='P2':
				visamp, visamp_err = self.visampSC_P2, self.visamperrSC_P2
				visphi, visphi_err = self.visphiSC_P2, self.visphierrSC_P2
			
			#vis2  , vis2_err   = self.vis2SC_P1  , self.vis2errSC_P1
			#t3    , t3_err     = self.t3SC_P1	 , self.t3errSC_P1 	
			
		#Perform fit		
		with multiprocessing.Pool(processes=n_cores, initializer=emcee_worker_init) as pool:
	
			sampler = emcee.EnsembleSampler(
				nwalkers=nwalkers, 
				ndim=ndim, 
				log_prob_fn=log_likelihood, 
				pool=pool,
				args=(parameters_to_fit,) )
			
			sampler.run_mcmc(
				initial_state=initial_emcee_state, 
				nsteps=steps, 
				progress=True)
		
		#Get the walker
		samples = sampler.get_chain(discard=20, thin=10, flat=True) 
		best_fit = np.median(samples, axis=0)  # Take the median of the posterior
		uncertainties = np.std(samples, axis=0)

		#Overwrite parameter 
		result_parameters = self.params.copy()

		for idx, elem in enumerate(parameters_to_fit):
			result_parameters[elem].value  = best_fit[idx]
			result_parameters[elem].stderr = uncertainties[idx]

		self.result_params = result_parameters

		self.sampler = sampler
		self.parameters_to_fit = parameters_to_fit

		return sampler, parameters_to_fit, result_parameters

			self,
			telescope_names,
			sources,
			background,
			l0=2.2,
			dl=0.2,
			reference_l0=2.2,
			use_phasemaps=True,
	):

		#Fetch baseline
		baseline_index = [idx for idx, x in enumerate(self.baseline_telescopes) if np.array_equal(x,telescope_names)][0]
		
		u = self.u[baseline_index]/units.micrometer
		v = self.v[baseline_index]/units.micrometer

		#Calculate things differently if using phasemaps or not
		if not use_phasemaps:
			
			#Storage variables
			visibility = 0.0
			normalization = 0.0

			for src in sources:

				#Get position, flux and spectral index for each source
				x, y, flux, alpha = src

				#Calculate optical path difference from position on sky and visibility
				s = (u*x + v*y)*units.mas_to_rad 

				#Add source visibility to nsource one
				visibility += flux*spectral_visibility(s, alpha, l0, dl, reference_l0)
				normalization += flux*spectral_visibility(0, alpha, l0, dl, reference_l0)

			#Add background to normalization
			flux, alpha = background
			normalization += flux*spectral_visibility(0, alpha, l0, dl,reference_l0)
			
			#Calculate spatial frequencies
			sf = np.sqrt(u**2+v**2)/l0*units.as_to_rad

			return sf, visibility/normalization

		else:
			
			#Storage variables
			visibility = 0.0
			normalization_i = 0.0
			normalization_j = 0.0

			#Telescope names
			tel_i, tel_j = telescope_names

			#Phasemaps
			#pms = self.phasemaps

			Ai = lambda sx, sy : self.phasemaps[telescope_to_beam[tel_i]]((l0,sx,sy))
			Aj = lambda sx, sy : self.phasemaps[telescope_to_beam[tel_j]]((l0,sx,sy))

			Li = lambda sx, sy : self.phasemaps_normalization[telescope_to_beam[tel_i]]((l0,sx,sy))
			Lj = lambda sx, sy : self.phasemaps_normalization[telescope_to_beam[tel_j]]((l0,sx,sy))

			for src in sources:

				#Get position, flux and spectral index for each source
				x, y, flux, alpha = src

				#Calculate optical path difference from position on sky and visibility
				s = (u*x + v*y)*units.mas_to_rad

				#Correct with phasemaps
				s -= ( Ai(x,y) - Aj(x,y))*l0/360 

				#Calculate visibility
				visibility += Ai(x,y)*Aj(x,y)*flux*spectral_visibility(s, alpha, l0, dl, reference_l0)
				normalization_i +=    Li(x,y)*flux*spectral_visibility(0, alpha, l0, dl, reference_l0)
				normalization_j +=    Lj(x,y)*flux*spectral_visibility(0, alpha, l0, dl, reference_l0)


		 	#Add background to normalization
			flux, alpha = background
			normalization_i += flux*spectral_visibility(0, alpha, l0, dl,reference_l0)
			normalization_j += flux*spectral_visibility(0, alpha, l0, dl,reference_l0)
    
			#Calculate spatial frequencies
			sf = np.sqrt(u**2+v**2)/l0*units.as_to_rad


			return sf, visibility/np.sqrt(normalization_i*normalization_j)
