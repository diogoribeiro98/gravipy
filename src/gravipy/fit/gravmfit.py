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

			model = GravMfit.nsource_visibility(
			uv_coordinates= [ucoord,vcoord],
			telescope_names = telescopes,
			sources=sources,
			background=background,
			l_list=self.wlSC,
			dl=self.dlambda,
			use_phasemaps=use_phasemaps,
			phasemaps= self.phasemaps,
			phasemaps_normalization=self.phasemaps_normalization
			)

			visibility_model[label] = model
		
		return visibility_model

	def nsource_visibility(
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
