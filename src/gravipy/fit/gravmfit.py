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
		self.use_phasemaps = None

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
		fit_sgr_pos = False,
		fit_sgr_fr  = False,
		fit_sgr_alpha = False,
		
		#Background parameters
		background_alpha = 3,
		background_fr = 0.1,
		fit_background_fr = False,
		fit_background_alpha = False,

		#Field type and fitting model
		field_type = 'star',
		fit_window_stars = None,
		fit_window_sgr = None,

		#Use phasemaps?
		use_phasemaps = True
						 ):
		
		#Check if the list of RA, Dec and Flux all have the same length
		if not all(len(lst) == len(ra_list) for lst in [de_list, fr_list]):
			raise  ValueError('RA, Dec and Flux lists must have the same length!')
		else:
			nsource = len(ra_list)

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
			self.use_phasemaps = use_phasemaps

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
					star_fit_parameters[f'source_{idx}_flux'] = [ 1.0, np.log10(0.001),  np.log10(100.), False ]
				else:
					star_fit_parameters[f'source_{idx}_flux'] = [ fr_list[idx]/fr_list[0],  np.log10(0.001),  np.log10(100.), fit_star_fr ]

			#Fitting parameters for background
			# number of parameters = 2
			background_fit_parameters = {
				'background_flux' : [background_fr   ,   0.0, 20.0, fit_background_fr   ],
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
				self.use_phasemaps = use_phasemaps

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
					'sgra_flux'  : [1.0			, np.log10(0.001), np.log10(100.), False        ],
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
				self.use_phasemaps = use_phasemaps

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
						star_fit_parameters[f'source_{idx}_flux'] = [ 1.0, np.log10(0.001),  np.log10(100.), False ]
					else:
						star_fit_parameters[f'source_{idx}_flux'] = [ fr_list[idx]/fr_list[0],  np.log10(0.001),  np.log10(100.), fit_star_fr ]

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
					'sgra_flux'  : [sgr_fr/fr_list[0]	, np.log10(0.001), np.log10(100.), fit_sgr_fr   ],
					'sgra_alpha' : [sgr_alpha			, -10.0	 		 , 10.0			 , fit_sgr_alpha],
				}

				#Fitting parameters for background
				# number of parameters = 2
				background_fit_parameters = {
					'background_flux' : [background_fr   ,  0.0, 20.0, fit_background_fr   ],
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
