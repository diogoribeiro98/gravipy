import os
import numpy as np

#Data processing tools
from astropy.io import fits
from scipy.signal import fftconvolve

#Gravipy tools
from ..physical_units import units 
from ..tools.time_and_dates import convert_date2
from ..tools.interferometric_beam import elliptical_beam_abc, estimate_npoints

from .dirty_beam import get_dirty_beam
from .interferometric_data_class import InterferometricData

#Plotting tools
import matplotlib.pyplot as plt
from ..tools.colors import colors_closure, colors_baseline

dualscivis_types = [
	'DUAL_SCI_VIS',
	'DUAL_SCI_VIS_CALIBRATED',
	]


class GraviData_scivis():
	"""Wrapper class for SCIVIS files. The class is called with a FITS file path as an argument:
	
	.. code-block:: python
		
		sci_data = GraviData_scivis("path/to/filename.FITS")

	All class variables can be acessed by ``sci_data.class_variable_name``.

	Class Variables:
		filepath (str): Absolute file path
		filename (str): Loaded filename
		header (fits.Header): FITS Header from file
		datacatg (str): Data Category (see section on :ref:`gravity_data`)
		datatype (str): Data Type (see section on :ref:`gravity_data`)
		date_obs (str)  : Date of observation as a string
		date (float): Date of observation in years
		mjd (float): Date of observation in MJD format
		polmode (str): Polarization mode 
		resolution (str): Observation resolution
		dit (int): Duration of each DIT
		ndit (int): Number of DITS
		tel (str): Array of telescopes used ('UT' or 'AT')
		object (str) : Name of observed object 
		ra (float) : Pointing RA 
		dec (float) : Pointing DEC
		sobj (str): Science object name
		sobj_x (float): RA offset from fringe tracker
		sobj_y (float): DEC offset from fringe tracker
		sobj_offx (float): RA offset in mas (from sobj_x)
		sobj_offy (float): DEC offset in mas (from sobj_y)
	"""

	def __init__(self, file):

		# ---------------------------
		# Pre-define class quantities
		# ---------------------------

		self.filepath : str | None
		self.filename : str | None

		self.header : fits.Header | None

		self.datacatg   : str | None
		self.datatype   : str | None

		self.date_obs   : str   | None
		self.date       : float | None
		self.mjd        : float | None

		self.polmode    : str 	| None 
		self.resolution : str 	| None
		self.dit        : float | None
		self.ndit       : int 	| None
		self.tel		: str   | None

		self.object : str   | None
		self.ra 	: float | None
		self.dec 	: float | None

		self.sobj   : float | None
		self.sobj_x : float | None
		self.sobj_y : float | None

		self.sobj_offx: float | None
		self.sobj_offy: float | None

		# ---------------------------
		# Check input file is ok
		# ---------------------------

		self.filepath =  os.path.abspath(file)
		self.filename =  os.path.basename(file)

		#Check if file exists
		if not os.path.exists(self.filepath):
			raise ValueError(f'File {self.filepath} does not exist')

		#Get header
		with fits.open(self.filepath) as hdul:
			self.header = hdul[0].header

		#Check it is a gravity file
		if self.header['INSTRUME'] != 'GRAVITY':
			raise ValueError('The input file is not a GRAVITY file.')

		#Check if UT or AT
		if self.header['TELESCOP'] in ['ESO-VLTI-U1234', 'U1234']:
			self.tel = 'UT'
		elif self.header['TELESCOP'] in ['ESO-VLTI-A1234', 'A1234']:
			self.tel = 'AT'
		else:
			raise ValueError('Telescope not AT or UT, seomtehign wrong with input data')
		
		#Check file type
		#Note: The keyword ESO PRO CATG stands for ESO PROCESSED CATEGORY!
		if 'ESO PRO CATG' in self.header:
			
			self.datacatg = self.header['ESO PRO CATG']
			self.datatype = self.header['ESO PRO TYPE']
			
			if self.datacatg not in dualscivis_types:
				raise ValueError(f'Filetype {self.datacatg} is not supported')
		
		else:
			raise ValueError(f'ESO PRO CATG not found. Perhaps you are loading a RAW file. Use the GraviData_raw for these files')

		# ---------------------------
		# Load Header data
		# ---------------------------

		#Time quantities
		self.date_obs   = self.header['DATE-OBS']
		self.date       = convert_date2(self.date_obs, mode='decimal')
		self.mjd        = convert_date2(self.date_obs, mode='mjd')
		self.date_arr   = convert_date2(self.date_obs, mode='split')

		#Intrumental mode and integration info
		self.polmode    = self.header['ESO INS POLA MODE']
		self.resolution = self.header['ESO INS SPEC RES']
		self.dit        = self.header['ESO DET2 SEQ1 DIT']
		self.ndit       = self.header['ESO DET2 NDIT']

		#Sky object and coordinates
		self.object = self.header['OBJECT']
		self.ra 	= self.header['RA']
		self.dec 	= self.header['DEC']

		#Field and pointing parameters
		self.sobj   = self.header['ESO INS SOBJ NAME']
		self.sobj_x = self.header['ESO INS SOBJ X'] # Distance from fringe tracker to field center in RA
		self.sobj_y = self.header['ESO INS SOBJ Y'] # Distance from fringe tracker to field center in DEC

		self.sobj_offx = self.header['ESO INS SOBJ OFFX'] #Distance between source field and current field in RA
		self.sobj_offy = self.header['ESO INS SOBJ OFFY'] #Distance between source field and current field in DEC
		
	#========================================
	# Data fetching functions
	#=========================================
	
	def get_interferometric_data(self, 
							  pol : str , 
							  channel : str ='SC', 
							  flag_channels : list = [],
							  *,
							  flag_channels_bl1 = [],
							  flag_channels_bl2 = [],
							  flag_channels_bl3 = [],
							  flag_channels_bl4 = [],
							  flag_channels_bl5 = [],
							  flag_channels_bl6 = [],
							  flag_channels_cl1 = [],
							  flag_channels_cl2 = [],
							  flag_channels_cl3 = [],
							  flag_channels_cl4 = [],
							  ):
		"""
		get_interferometric_data(pol, channel='SC', flag_channels=[], **kwargs)

		Returns instance of InterferometricData.

		Args:
			pol (str): Polarization to retrieve. Must be 'P1' or 'P2'.
			channel (str, optional): Channel to retrieve. Must be 'SC'(science) or 'FT'(fringe tracker). Defaults to 'SC'.
			flag_channels (list, optional): Wavelenght channels to flag. Defaults to [].
		
		Keyword Args:
			flag_channels_bl1 (list): Channels to flag on baseline 1.
			flag_channels_bl2 (list): Channels to flag on baseline 2.
			flag_channels_bl3 (list): Channels to flag on baseline 3.
			flag_channels_bl4 (list): Channels to flag on baseline 4.
			flag_channels_bl5 (list): Channels to flag on baseline 5.
			flag_channels_bl6 (list): Channels to flag on baseline 6.
			flag_channels_cl1 (list): Channels to flag on closure triangle 1.
			flag_channels_cl2 (list): Channels to flag on closure triangle 2.
			flag_channels_cl3 (list): Channels to flag on closure triangle 3.
			flag_channels_cl4 (list): Channels to flag on closure triangle 4.

		Returns:
			InterferometricData: interferometric data class
		"""

		if self.polmode=='COMBINED':
			raise ValueError('COMBINED polarization is currently not supported. Load a SPLIT file.')

		if pol not in ['P1', 'P2']:
			raise ValueError(f"{pol} is not a valid polarization. Must be ['P1', 'P2'].")

		if channel not in ['SC', 'FT']:
			raise ValueError(f"{channel} is not a valid channel. Must be ['SC', 'FT'].")

		# Set fits index for each combination of input parameters
		# Note: The following lines concretize the following mapping:
		# {channel, polarization} -> 11, 12, 21, 22
		# depending on the channel and polarization
		fits_index = int(f"{1 if channel == 'SC' else 2}{1 if pol == 'P1' else 2}")

		with fits.open(self.filepath) as f:

			#Fetch data fields
			oi_array		= f['OI_ARRAY'].data
			oi_wavelenght 	= f['OI_WAVELENGTH', fits_index].data			
			oi_vis 			= f['OI_VIS'	   , fits_index].data
			oi_vis2  		= f['OI_VIS2'	   , fits_index].data
			oi_t3  			= f['OI_T3'	  	   , fits_index].data
			oi_flux			= f['OI_FLUX' 	   , fits_index].data
			
			#---------------------------------
			# Fetch telescope array quantities
			#---------------------------------

			#Baseline coordinates in meters
			Bu, Bv = oi_vis['UCOORD'], oi_vis['VCOORD']
			
			#Telescope names and indices
			sta_names = oi_array['TEL_NAME']
			sta_pos   = oi_array['STAXYZ'][:,0:2]
			sta_index = oi_array['STA_INDEX']

			sta_index_to_name = dict(zip(sta_index, sta_names))

			#Baseline and Closure indices
			bl_index = oi_vis['STA_INDEX']
			t3_index  = oi_t3['STA_INDEX']

			bl_telescopes 	= np.empty((len(bl_index),2), dtype=object)
			bl_labels 		= np.empty(len(bl_index)	, dtype=object)
			t3_telescopes 	= np.empty((len(t3_index),3), dtype=object)
			t3_labels 		= np.empty(len(t3_index)	, dtype=object)

			for idx,bl in enumerate(bl_index):
				t1 = sta_index_to_name[bl[0]]
				t2 = sta_index_to_name[bl[1]]
				bl_telescopes[idx]  = np.array([t1,t2])
				bl_labels[idx] 		= t1+t2[-1]
			
			for idx,cl in enumerate(t3_index):
				t1 = sta_index_to_name[cl[0]]
				t2 = sta_index_to_name[cl[1]]
				t3 = sta_index_to_name[cl[2]]
				t3_telescopes[idx] = np.array([t1,t2,t3])
				t3_labels[idx] 		= t1+t2[-1]+t3[-1]
			
			#--------------------------------------
			# Fetch wavelength quantities and flux
			#--------------------------------------

			#Fetch wavelength quantities
			wave =  oi_wavelenght['EFF_WAVE']/units.micrometer
			band = (oi_wavelenght['EFF_BAND']/units.micrometer)/2
			
			#Fetch flux quantities
			flux, flux_err = oi_flux['FLUX'], oi_flux['FLUXERR']

			#---------------------------------
			# Fetch interferometric data
			#---------------------------------

			#Amplitude and Phase
			visamp, visamp_err = oi_vis['VISAMP'], oi_vis['VISAMPERR']  
			visphi, visphi_err = oi_vis['VISPHI'], oi_vis['VISPHIERR'] 

			vis_flag = oi_vis['FLAG']
			
			#Amplitude squared
			vis2  , vis2_err, vis2_flag = oi_vis2['VIS2DATA'], oi_vis2['VIS2ERR'], oi_vis2['FLAG'] 

			#Closure Phase
			t3amp  , t3amp_err= oi_t3['T3AMP'], oi_t3['T3AMPERR'] 
			t3phi  , t3phi_err= oi_t3['T3PHI'], oi_t3['T3PHIERR'] 

			t3_flag = oi_t3['FLAG']
			
			#
			# Clean up flagged chanels			
			#

			vis_flag[visamp>1.0] 		= 1
			vis_flag[np.isnan(visamp)]	= 1
			vis_flag[np.isnan(visphi)] 	= 1
			vis_flag[:,flag_channels] 	= 1

			vis2_flag[vis2>1.0] 		= 1
			vis2_flag[vis2<0.0] 		= 1
			vis2_flag[np.isnan(vis2)]   = 1
			vis2_flag[:,flag_channels] 	= 1

			t3_flag[np.isnan(t3amp)] = 1
			t3_flag[np.isnan(t3phi)] = 1
			t3_flag[:,flag_channels] = 1

			#
			# Check extra flagged channels
			#

			vis_flag[0,flag_channels_bl1] = 1
			vis_flag[1,flag_channels_bl2] = 1
			vis_flag[2,flag_channels_bl3] = 1
			vis_flag[3,flag_channels_bl4] = 1
			vis_flag[4,flag_channels_bl5] = 1
			vis_flag[5,flag_channels_bl6] = 1

			vis2_flag[0,flag_channels_bl1] = 1
			vis2_flag[1,flag_channels_bl2] = 1
			vis2_flag[2,flag_channels_bl3] = 1
			vis2_flag[3,flag_channels_bl4] = 1
			vis2_flag[4,flag_channels_bl5] = 1
			vis2_flag[5,flag_channels_bl6] = 1

			t3_flag[0,flag_channels_cl1] = 1
			t3_flag[1,flag_channels_cl2] = 1
			t3_flag[2,flag_channels_cl3] = 1
			t3_flag[3,flag_channels_cl4] = 1

			#
			# Flag channels
			#

			for data in [visamp,visamp_err, visphi, visphi_err]:				
				data[vis_flag]   = np.nan

			for data in [vis2, vis2_err]:
				data[vis2_flag] = np.nan
			
			for data in [t3amp, t3amp_err, t3phi, t3phi_err]:
				data[t3_flag] = np.nan

			#-------------------------------------------
			# Fetch baseline and spatial frequency info
			#-------------------------------------------

			Bu, Bv = oi_vis['UCOORD'], oi_vis['VCOORD']
			B_magnitude = np.sqrt(Bu**2.+Bv**2.)

			u = np.zeros((len(Bu), len(oi_wavelenght)))
			v = np.zeros((len(Bv), len(oi_wavelenght)))

			for i in range(0, len(Bu)):
				u[i, :] = (Bu[i] / (wave * units.micrometer)) / units.rad_to_as # 1/as
				v[i, :] = (Bv[i] / (wave * units.micrometer)) / units.rad_to_as # 1/as
			
			spatial_frequency_as = np.sqrt(u**2.+v**2.)

		 	#Note: For closure phases, it is custumary to use as reference the longest baseline
			max_B = np.zeros(4)

			max_B[0] = np.max(np.array([B_magnitude[0], B_magnitude[3], B_magnitude[1]]))
			max_B[1] = np.max(np.array([B_magnitude[0], B_magnitude[4], B_magnitude[2]]))
			max_B[2] = np.max(np.array([B_magnitude[1], B_magnitude[5], B_magnitude[2]]))
			max_B[3] = np.max(np.array([B_magnitude[3], B_magnitude[5], B_magnitude[4]]))

			spatial_frequency_as_T3 = np.zeros((len(max_B), len(wave)))
			
			for idx in range(len(max_B)):
				spatial_frequency_as_T3[idx] = ( max_B[idx] / (wave * units.micrometer) ) / units.rad_to_as  # 1/as

			data = InterferometricData(
				filename=self.filename,
				header = self.header,
				date_obs=self.date_obs,
				polmode = self.polmode,
				resolution = self.resolution,
				object = self.object,
				ra	= self.ra, 	
				dec = self.dec, 
				sobj   = self.sobj,
				sobj_x = self.sobj_x,
				sobj_y = self.sobj_y,
				sobj_offx = self.sobj_offx,
				sobj_offy = self.sobj_offy,
				pol=pol,
				Bu=Bu, Bv=Bv, 
				telescopes=sta_names,
				tel_pos=sta_pos,
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
				spatial_frequency_as=spatial_frequency_as,
				spatial_frequency_as_T3=spatial_frequency_as_T3,
				flux=flux, flux_err=flux_err
			)

			return data

	def get_dirty_beam(
			self,
			pol,
			channel='SC', 
			flag_channels=[],
			*,
			window=75,
			gain=1.0, 
			threshold=1e-3, 
			max_iter=None,
			pixels_per_beam = 2,
			flag_channels_bl1 = [],
			flag_channels_bl2 = [],
			flag_channels_bl3 = [],
			flag_channels_bl4 = [],
			flag_channels_bl5 = [],
			flag_channels_bl6 = [],
			flag_channels_cl1 = [],
			flag_channels_cl2 = [],
			flag_channels_cl3 = [],
			flag_channels_cl4 = [],
			):
		"""		
		get_dirty_beam(pol, channel='SC', flag_channels=[], **kwargs)

		Returns the Dirty Beam, Dirty Image and Clean Map of interferometric data

		Args:
			pol (str): polarization to retrieve. Must be 'P1' or 'P2'.
			channel (str): channel to retrieve. Must be 'SC'(science) or 'FT'(fringe tracker) Defaults to 'SC'.
			flag_channels (list, optional): Wavelenght channels to flag. Defaults to [].
		Keyword Args:
			window (int, optional): Size of reconstructed image in miliarcseconds. Defaults to 80.
			gain (float, optional): CLEAN loop gain. Defaults to 1.0.
			threshold (float, optional): CLEAN loop threshold. Defaults to 0.05.
			max_iter (int, optional): Maxium CLEAN iterations. If None is give, uses the number of UV sampled points. Defaults to None.
			pixels_per_beam (int, optional): Size of central beam in pixels. Defaults to 2.
			flag_channels_bl1 (list): Channels to flag on baseline 1.
			flag_channels_bl2 (list): Channels to flag on baseline 2.
			flag_channels_bl3 (list): Channels to flag on baseline 3.
			flag_channels_bl4 (list): Channels to flag on baseline 4.
			flag_channels_bl5 (list): Channels to flag on baseline 5.
			flag_channels_bl6 (list): Channels to flag on baseline 6.
			flag_channels_cl1 (list): Channels to flag on closure triangle 1.
			flag_channels_cl2 (list): Channels to flag on closure triangle 2.
			flag_channels_cl3 (list): Channels to flag on closure triangle 3.
			flag_channels_cl4 (list): Channels to flag on closure triangle 4.

		Returns:
			Dirty Beam, Dirty Image and Clean map in the format (x, 2D Array)		
			
			(bx, B.real) : Dirty Beam 
			
			(x, I.real)  : Dirty Image 
			
			(x,clean_map.real): Clean Map
		
		"""		

		#Fetch data
		idata = self.get_interferometric_data(
			pol,
			channel,
			flag_channels,
			flag_channels_bl1 = flag_channels_bl1,
			flag_channels_bl2 = flag_channels_bl2,
			flag_channels_bl3 = flag_channels_bl3,
			flag_channels_bl4 = flag_channels_bl4,
			flag_channels_bl5 = flag_channels_bl5,
			flag_channels_bl6 = flag_channels_bl6,
			flag_channels_cl1 = flag_channels_cl1,
			flag_channels_cl2 = flag_channels_cl2,
			flag_channels_cl3 = flag_channels_cl3,
			flag_channels_cl4 = flag_channels_cl4,
			)
		
		(xb, B), (x, I), (x,I) =  get_dirty_beam(
			idata,
			window=window,
			gain=gain, 
			threshold=threshold, 
			max_iter=max_iter,
			pixels_per_beam = pixels_per_beam
		)

		return (xb, B), (x, I), (x,I)

	#========================================
	# Plotting functions
	#=========================================
	
	def plot_interferometric_data(
			self, 
			pol='P1', 
			channel='SC', 
			flag_channels=[],
			*,
			flag_channels_bl1 = [],
			flag_channels_bl2 = [],
			flag_channels_bl3 = [],
			flag_channels_bl4 = [],
			flag_channels_bl5 = [],
			flag_channels_bl6 = [],
			flag_channels_cl1 = [],
			flag_channels_cl2 = [],
			flag_channels_cl3 = [],
			flag_channels_cl4 = [],
			):
		"""
		plot_interferometric_data(pol, channel='SC', flag_channels=[], **kwargs)

		Plot all interferometric quantities

		Args:
			pol (str): polarization to retrieve. Must be 'P1' or 'P2'.
			channel (str): channel to retrieve. Must be 'SC'(science) or 'FT'(fringe tracker) Defaults to 'SC'.
			flag_channels (list, optional): Wavelenght channels to flag. Defaults to [].

		Keyword Args:
			flag_channels_bl1 (list): Channels to flag on baseline 1.
			flag_channels_bl2 (list): Channels to flag on baseline 2.
			flag_channels_bl3 (list): Channels to flag on baseline 3.
			flag_channels_bl4 (list): Channels to flag on baseline 4.
			flag_channels_bl5 (list): Channels to flag on baseline 5.
			flag_channels_bl6 (list): Channels to flag on baseline 6.
			flag_channels_cl1 (list): Channels to flag on closure triangle 1.
			flag_channels_cl2 (list): Channels to flag on closure triangle 2.
			flag_channels_cl3 (list): Channels to flag on closure triangle 3.
			flag_channels_cl4 (list): Channels to flag on closure triangle 4.

		Returns:
			(fig, ax) : Figure and axes
		"""

		#Fetch data
		idata = self.get_interferometric_data(
			pol,
			channel,
			flag_channels = flag_channels,
			flag_channels_bl1 = flag_channels_bl1,
			flag_channels_bl2 = flag_channels_bl2,
			flag_channels_bl3 = flag_channels_bl3,
			flag_channels_bl4 = flag_channels_bl4,
			flag_channels_bl5 = flag_channels_bl5,
			flag_channels_bl6 = flag_channels_bl6,
			flag_channels_cl1 = flag_channels_cl1,
			flag_channels_cl2 = flag_channels_cl2,
			flag_channels_cl3 = flag_channels_cl3,
			flag_channels_cl4 = flag_channels_cl4,
			)

	 	#Define helper plot configurations
		plot_config = {
			'alpha':    0.8,
			'ms':       3.0,
			'lw':       0.8,
			'capsize':  1.0,
			'ls':       ''    
		}
		
		fig, axes = plt.subplots(ncols=5, figsize=(5*4.2,3))

		#Visibility Amplitude plot
		ax = axes[0]
		ax.set_title('Visibility Amplitude')
		ax.set_xlim(70, 320)
		ax.set_ylim(-0.1, 1.1)
		ax.set_ylabel('Visibility Amplitude')
		ax.set_xlabel('spatial frequency (1/arcsec)')
		ax.axhline(1, ls='--', lw=0.8, c='k' )
		ax.axhline(0, ls='--', lw=0.8, c='k' )
		ax.grid()

		for idx, x in enumerate(idata.spatial_frequency_as):
			y    = idata.visamp[idx] 
			yerr = idata.visamp_err[idx] 
			ax.errorbar(x, y, yerr, **plot_config, marker='o', color=colors_baseline[idx])
		
		#Visibility Phase plot
		ax = axes[1]
		ax.set_title('Visibility Phase')
		ax.set_xlim(70, 320)
		ax.set_ylim(-200, 200)
		ax.set_ylabel('Visibility Phase')
		ax.set_xlabel('spatial frequency (1/arcsec)')
		ax.axhline(180, ls='--', lw=0.8, c='k' )
		ax.axhline(-180, ls='--', lw=0.8, c='k' )
		ax.grid()

		for idx, x in enumerate(idata.spatial_frequency_as):
			y    = idata.visphi[idx] 
			yerr = idata.visphi_err[idx] 
			ax.errorbar(x, y, yerr, **plot_config, marker='o', color=colors_baseline[idx])
		
		#Visibility Squared plot
		ax = axes[2]
		ax.set_title('Visibility Squared')
		ax.set_xlim(70, 320)
		ax.set_ylim(-0.1, 1.1)
		ax.set_ylabel('Visibility Squared')
		ax.set_xlabel('spatial frequency (1/arcsec)')
		ax.axhline(1, ls='--', lw=0.8, c='k' )
		ax.axhline(0, ls='--', lw=0.8, c='k' )
		ax.grid()

		for idx, x in enumerate(idata.spatial_frequency_as):
			y    = idata.vis2[idx] 
			yerr = idata.vis2_err[idx] 
			ax.errorbar(x, y, yerr, **plot_config, marker='o', color=colors_baseline[idx])
		
		#Closure Amplitudes
		ax = axes[3]
		ax.set_title('Closure Amplitude')
		ax.set_xlim(150, 320)
		ax.set_ylim(-0.1, 1.1)
		ax.set_ylabel('Closure Amplitude')
		ax.set_xlabel('spatial frequency (1/arcsec)')
		ax.axhline(1, ls='--', lw=0.8, c='k' )
		ax.axhline(0, ls='--', lw=0.8, c='k' )
		ax.grid()

		for idx, x in enumerate(idata.spatial_frequency_as_T3):
			y    = idata.t3amp[idx] 
			yerr = idata.t3amp_err[idx] 
			ax.errorbar(x, y, yerr, **plot_config, marker='o', color=colors_closure[idx])
		
		#Closure phases
		ax = axes[4]
		ax.set_title('Closure Phases')
		ax.set_xlim(150, 320)
		ax.set_ylim(-200, 200)
		ax.set_ylabel('Closure Phases')
		ax.set_xlabel('spatial frequency (1/arcsec)')
		ax.axhline(180, ls='--', lw=0.8, c='k' )
		ax.axhline(-180, ls='--', lw=0.8, c='k' )
		ax.grid()
		
		for idx, x in enumerate(idata.spatial_frequency_as_T3):
			y    = idata.t3phi[idx] 
			yerr = idata.t3phi_err[idx] 
			ax.errorbar(x, y, yerr, **plot_config, marker='o', color=colors_closure[idx])
		
		return fig, axes

	def plot_dirty_beam(
			self,
			pol,
			channel='SC', 
			flag_channels=[],
			*,
			window=75,
			gain=1.0, 
			threshold=1e-3, 
			max_iter=None,
			pixels_per_beam=2,
			flag_channels_bl1 = [],
			flag_channels_bl2 = [],
			flag_channels_bl3 = [],
			flag_channels_bl4 = [],
			flag_channels_bl5 = [],
			flag_channels_bl6 = [],
			flag_channels_cl1 = [],
			flag_channels_cl2 = [],
			flag_channels_cl3 = [],
			flag_channels_cl4 = [],
			cmap='gist_yarg',
			):
		"""		
		plot_dirty_beam(pol, channel='SC', flag_channels=[], **kwargs)

		PLots the Dirty Beam, Dirty Image and Clean Map of interferometric data

		Args:
			pol (str): polarization to retrieve. Must be 'P1' or 'P2'.
			channel (str): channel to retrieve. Must be 'SC'(science) or 'FT'(fringe tracker) Defaults to 'SC'.
			flag_channels (list, optional): Wavelenght channels to flag. Defaults to [].
		Keyword Args:
			window (int, optional): Size of reconstructed image in miliarcseconds. Defaults to 80.
			gain (float, optional): CLEAN loop gain. Defaults to 1.0.
			threshold (float, optional): CLEAN loop threshold. Defaults to 0.05.
			max_iter (int, optional): Maxium CLEAN iterations. If None is give, uses the number of UV sampled points. Defaults to None.
			pixels_per_beam (int, optional): Size of central beam in pixels. Defaults to 2.
			flag_channels_bl1 (list): Channels to flag on baseline 1.
			flag_channels_bl2 (list): Channels to flag on baseline 2.
			flag_channels_bl3 (list): Channels to flag on baseline 3.
			flag_channels_bl4 (list): Channels to flag on baseline 4.
			flag_channels_bl5 (list): Channels to flag on baseline 5.
			flag_channels_bl6 (list): Channels to flag on baseline 6.
			flag_channels_cl1 (list): Channels to flag on closure triangle 1.
			flag_channels_cl2 (list): Channels to flag on closure triangle 2.
			flag_channels_cl3 (list): Channels to flag on closure triangle 3.
			flag_channels_cl4 (list): Channels to flag on closure triangle 4.

		Plot Arguments:
			cmap (str): Matplotlib colormap
		
		Returns:
			(fig, ax) : Figure and axes
		"""

		(bx,B), (x,I), (x,C) = self.get_dirty_beam(
			pol=pol,
			channel=channel, 
			flag_channels=flag_channels,
			window=window,
			gain=gain, 
			threshold=threshold, 
			max_iter=max_iter,
			pixels_per_beam=pixels_per_beam,
			flag_channels_bl1 = flag_channels_bl1,
			flag_channels_bl2 = flag_channels_bl2,
			flag_channels_bl3 = flag_channels_bl3,
			flag_channels_bl4 = flag_channels_bl4,
			flag_channels_bl5 = flag_channels_bl5,
			flag_channels_bl6 = flag_channels_bl6,
			flag_channels_cl1 = flag_channels_cl1,
			flag_channels_cl2 = flag_channels_cl2,
			flag_channels_cl3 = flag_channels_cl3,
			flag_channels_cl4 = flag_channels_cl4,
			)
		

		#Create figure
		fig, axes = plt.subplots(ncols=3, figsize=(11,3), dpi = 300)

		axes[0].set_title('Dirty Beam')
		axes[1].set_title('Dirty Image')
		axes[2].set_title('Clean Image')

		axes[0].pcolormesh(bx,bx, B.real, 	cmap=cmap)
		axes[1].pcolormesh(x,  x, I.real,   cmap=cmap)
		axes[2].pcolormesh(x , x, C,		cmap=cmap)
		#cb = plt.colorbar(im)

		fiber_fov=70

		for ax in axes:
			circ = plt.Circle((0,0), radius=fiber_fov, facecolor="None", edgecolor='black', linewidth=0.8)
			ax.add_artist(circ)
			ax.set_xlim(-fiber_fov*1.05,fiber_fov*1.05)
			ax.set_ylim(-fiber_fov*1.05,fiber_fov*1.05)
			ax.invert_xaxis()
			ax.set_aspect(1)

		return fig, axes
	
	def plot_spectral_flux(
			self,
			pol,
			channel='SC'):
		
		#Fetch data
		idata = self.get_interferometric_data(pol, channel)
		
		x 	 = idata.wave
		y 	 = idata.flux
		yerr = idata.flux_err
		
		fig, ax = plt.subplots(figsize=(5,2.2))
		ax.set_xlabel('Wavelength [micrometer]')
		ax.set_ylabel(fr'Flux [a.u.]')
		ax.set_xlim(1.98,2.5)
		ax.axvline(2.0, ls='-.', lw=0.8,c='k')
		ax.axvline(2.4, ls='-.', lw=0.8,c='k')
		#ax.set_ylim(-2,2)

		for idx, tel in enumerate(idata.telescopes):
			ax.errorbar(x,y[idx],yerr=yerr[idx])

		return fig, ax