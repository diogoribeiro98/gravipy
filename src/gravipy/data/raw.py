import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mplc

from astropy.io import fits

from ..tools.time_and_dates import convert_date2

raw_types = [			
'OBJECT,DUAL',
'OBJECT,SINGLE',
'SKY,DUAL',
'SKY,SINGLE'
]

class GraviData_raw():
	"""Wrapper class to deal with raw files
	"""

	def __init__(self, file):
			
		# ---------------------------
		# Pre-define class quantities
		# ---------------------------

		self.filepath : str | None  #: Absolute file path
		self.filename : str | None  #: Loaded file name

		self.header : fits.Header | None #: FITS Header

		self.datacatg   : str | None #: Data Category (see :ref:`gravipy`)
		self.datatype   : str | None

		self.date_obs   : str | None
		self.date       : str | None
		self.mjd        : str | None

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
		#Note: The keyword ESO DPR CATG stands for ESO DATA PRODUCT CATEGORY!
		if 'ESO DPR CATG' in self.header:
			
			self.datacatg = self.header['ESO DPR CATG']
			self.datatype = self.header['ESO DPR TYPE']
			
			if self.datacatg not in ['SCIENCE', 'CALIB']:
				raise ValueError(f'Raw data product with category {self.datacatg} is not supported')
	
		else:
			raise ValueError(f'ESO DPR CATG not found. Perhaps you are NOT loading a RAW file.')

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
	
	def get_acq_data(self):
		"""Get aquisition camera data

		Returns:
			_type_: _description_
		"""
		with fits.open(self.filepath) as f:
			aqc_data = f['IMAGING_DATA_ACQ'].data
		return aqc_data
	
	def get_sc_data(self):
		with fits.open(self.filepath) as f:
			sc_data = f['IMAGING_DATA_SC'].data
		return sc_data
	
	def get_ft_data(self):
		with fits.open(self.filepath) as f:
			ft_data = f['IMAGING_DATA_FT'].data
		return ft_data
	
	#========================================
	# Aquisition camera display functions
	#=========================================

	def plot_acq_beacons(self, vmin =-500, vmax = 2500):
		"""Plot the frame average of the laser beacons
		"""
		
		#Get beacon data
		acq_data = self.get_acq_data()
		frame_data = np.average(acq_data[:,750:1000,:], axis=0)

		#Setup figure
		fig, ax = plt.subplots( figsize=(4,2.5))	
		ax.set_title(self.filename, fontsize=6)	
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		ax.imshow(
			frame_data,
			cmap='gist_heat', 
			origin='lower',
			vmin=vmin,
			vmax=vmax
			)

		for idx, beam in enumerate(['GV1', 'GV2', 'GV3', 'GV4']):
			ax.text(x = idx*250+10, y = 215, s = beam, c='#22ff00', fontsize=6)
		
		return fig, ax
	
	def plot_acq_fields(self, percentile=99.6):
		"""Plot the frame average of the acquisition fields
		"""
		
		#Get beacon data
		acq_data = self.get_acq_data()
		frame_data = np.average(acq_data[:,0:250,:], axis=0)

		#Setup figure
		fig, ax = plt.subplots( figsize=(4,2.5))	
		ax.set_title(self.filename, fontsize=6)	
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		lower_limit = np.percentile(frame_data, 0.1)
		upper_limit = np.percentile(frame_data, percentile)

		ax.imshow(
			frame_data,
			cmap='gist_heat', 
			origin='lower',
			norm=mplc.AsinhNorm(vmin=lower_limit, vmax=upper_limit)
			)

		for idx, beam in enumerate(['GV1', 'GV2', 'GV3', 'GV4']):
			ax.text(x = idx*250+10, y = 215, s = beam, c='#22ff00', fontsize=6)
	
		return fig, ax
	
	def plot_acq_camera(self, percentile=99.6):
		"""	Plot the frame average of the acquisition camera
		"""

		acq_data = self.get_acq_data()
		frame_data = np.average(acq_data, axis=0)

		fig, ax = plt.subplots(figsize=(5,5))
		ax.set_title(self.filename, fontsize=6)	

		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		lower_limit = 1
		upper_limit = np.percentile(frame_data[0:250,:], percentile)

		ax.imshow(
			frame_data,
			cmap='gist_heat', 
			origin='lower',
			norm=mplc.AsinhNorm(vmin=lower_limit, vmax=upper_limit)
			)
		
		for idx, beam in enumerate(['GV1', 'GV2', 'GV3', 'GV4']):
			ax.text(x = idx*250+10, y = 1000-25, s = beam, c='#22ff00', fontsize=6)

		return fig, ax

	#========================================
	# Science camera display functions
	#=========================================

	def plot_sc_data(self, vmin =0, vmax = 250):
		"""Plot the frame average of the laser beacons
		"""
		
		#Get beacon data
		sc_data = self.get_sc_data()
		frame_data = np.average(sc_data, axis=0)

		#Setup figure
		fig, ax = plt.subplots()	
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xticklabels([])
		ax.set_yticklabels([])

		ax.imshow(
			frame_data,
			cmap='gist_heat', 
			origin='lower',
			vmin=vmin,
			vmax=vmax
			)

		return fig, ax
