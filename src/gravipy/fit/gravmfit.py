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
