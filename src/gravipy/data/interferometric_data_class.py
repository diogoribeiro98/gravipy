import numpy as np
from astropy.io import fits
import h5py
import os

from dataclasses import dataclass

@dataclass
class InterferometricData:
	"""
	A class to hold the interferometric data and associated metadata, including 
	header information, observation details, polarization, metrology corrections, 
	baseline data, telescope positions, visibility data (amplitude, phase), and more.

	Besides the explicit class members, several fetching methods are implemented to quickly access important header variables.

	Further more, the interferometric data can be saved or loaded from an hdf file.

	Parameters:
		filename (str): The name of the data file.
		header (fits.Header): Main header of file
		date_obs (str): The date and time of observation.
		polmode (str): Observation polarization mode
		resolution(str): Observation resolution
		object (str): The astronomical object being observed.
		ra (float): The right ascension of the observed object in degrees.
		dec (float): The declination of the observed object in degrees.
		sobj (float): The source object flux density.
		sobj_x (float): The source object flux density in the x-direction.
		sobj_y (float): The source object flux density in the y-direction.
		sobj_offx (float): The source object offset in the x-direction.
		sobj_offy (float): The source object offset in the y-direction.
		pol (str): The polarization type of the observation (e.g., 'linear', 'circular').
		Bu (np.ndarray): Baseline vector components in the u-direction.
		Bv (np.ndarray): Baseline vector components in the v-direction.
		telescopes (np.ndarray): Telescopes' names.
		tel_pos (np.ndarray): The position of the telescope(s) during the observation.
		bl_telescopes (np.ndarray): List of telescopes used for baseline measurements.
		t3_telescopes (np.ndarray): List of telescopes used for closure phase measurements.
		bl_labels (np.ndarray): Labels for the baselines.
		t3_labels (np.ndarray): Labels for the closure phase baselines.
		wave (np.ndarray): The wavelength(s) of observation.
		band (np.ndarray): The bandwidth(s) of the observation.
		flux (np.ndarray): Flux per wavelength
		flux_err (np.ndarray): Flux error
		visamp (np.ndarray): Visibility amplitude data.
		visamp_err (np.ndarray): Visibility amplitude error.
		visphi (np.ndarray): Visibility phase data.
		visphi_err (np.ndarray): Visibility phase error.
		vis_flag (np.ndarray): Visibility flag to indicate valid or invalid data.
		vis2 (np.ndarray): Squared visibility data.
		vis2_err (np.ndarray): Squared visibility error.
		vis2_flag (np.ndarray): Flag for squared visibility data.
		t3amp (np.ndarray): Closure amplitude data.
		t3amp_err (np.ndarray): Closure amplitude error.
		t3phi (np.ndarray): Closure phase data.
		t3phi_err (np.ndarray): Closure phase error.
		t3flag (np.ndarray): Flag for closure phase data.
		spatial_frequency_as (np.ndarray): Spatial frequency in arcseconds for baseline data.
		spatial_frequency_as_T3 (np.ndarray): Spatial frequency in arcseconds for closure phase data.
	"""

	#--------------------------
	# General file information
	#--------------------------

	#File and Header information
	filename: str
	header : fits.Header
	
	#Obsevation information
	date_obs: str
	polmode    : str 
	resolution : str
		
	#Observation pointing
	object 	: str
	ra 		: float
	dec 	: float

	sobj   : float
	sobj_x : float
	sobj_y : float

	sobj_offx: float 
	sobj_offy: float 

	#--------------------------
	# Interferometric data
	#--------------------------

	#Polarization
	pol: str

	#Baselines
	Bu: np.ndarray
	Bv: np.ndarray

	#Telescope position
	telescopes: np.ndarray
	tel_pos: 	np.ndarray

	#Baseline and closure telescopes
	bl_telescopes: np.ndarray
	t3_telescopes: np.ndarray
	bl_labels: np.ndarray
	t3_labels: np.ndarray

	#Wavelength
	wave: np.ndarray
	band: np.ndarray

	#Flux
	flux: np.ndarray
	flux_err: np.ndarray

	#Visibility amplitude and phase
	visamp: np.ndarray
	visamp_err: np.ndarray
	visphi: np.ndarray
	visphi_err: np.ndarray
	vis_flag: np.ndarray

	#Visibility squared
	vis2: np.ndarray
	vis2_err: np.ndarray
	vis2_flag: np.ndarray

	#Closure amplitude and phase
	t3amp: np.ndarray
	t3amp_err: np.ndarray
	t3phi: np.ndarray
	t3phi_err: np.ndarray
	t3flag: np.ndarray

	#Spacial frequencies
	spatial_frequency_as: np.ndarray
	spatial_frequency_as_T3: np.ndarray

	def get_metrology_offset_correction(self):
		""" Returns metrology offset correction
		"""
		
		sobj_metrology_correction_x ={
			'GV1': self.header['ESO QC MET SOBJ DRA1'], 
			'GV2': self.header['ESO QC MET SOBJ DRA2'],
			'GV3': self.header['ESO QC MET SOBJ DRA3'],
			'GV4': self.header['ESO QC MET SOBJ DRA4'],
		}

		sobj_metrology_correction_y ={
			'GV1': self.header['ESO QC MET SOBJ DDEC1'], 
			'GV2': self.header['ESO QC MET SOBJ DDEC2'],
			'GV3': self.header['ESO QC MET SOBJ DDEC3'],
			'GV4': self.header['ESO QC MET SOBJ DDEC4'],
		}

		return sobj_metrology_correction_x, sobj_metrology_correction_y

	def get_acq_north_angle(self):
		""" Return estimated north angle values estimated from aquisition camera

		.. important::
			
			If the loaded data was not reduced reduced with the ``-reduce-acq-cam=TRUE`` as an argument for the ``gravi_vis`` pipeline recipe, the function will raise an error.

		"""
		try:
			north_angle = {
				'GV1': self.header['ESO QC ACQ FIELD1 NORTH_ANGLE']*np.pi/180., 
				'GV2': self.header['ESO QC ACQ FIELD2 NORTH_ANGLE']*np.pi/180.,
				'GV3': self.header['ESO QC ACQ FIELD3 NORTH_ANGLE']*np.pi/180.,
				'GV4': self.header['ESO QC ACQ FIELD4 NORTH_ANGLE']*np.pi/180.,
				}
		except:
			raise ValueError(''\
			'NORTH ANGLE parameters not found in header. ' \
			'Make sure your SCIVIS file was reduced using the -reduce-acq-cam=TRUE'
			)

		return north_angle

'''
	def to_hdf5(self, filepath,*, mode='write'):

		
		valid_modes = ('write', 'append', 'overwrite')
		if mode not in valid_modes:
			raise ValueError(f"mode must be one of {valid_modes}")
		
		#Save behaviour depends on file existence
		file_exists = os.path.exists(filepath)

		if mode == 'write':
			if file_exists:
				raise FileExistsError(f"File '{filepath}' already exists. Use 'overwrite_file' if you want to replace it.")
			else:
				file_mode = 'w'

		elif mode == 'append':
			if not file_exists:
				raise FileNotFoundError(f"File '{filepath}' does not exist. Use 'write_file' to create it.")
			else:
				file_mode = 'a'

		elif mode =='overwrite':



		if mode not in ('write_file', 'append_to_file'):
			raise ValueError("mode must be 'write_file' or 'append_to_file'")
	
		file_mode = 'w' if mode == 'write_file' else 'a'
		group_name = 'idata'

		if os.path.exists(filepath) and not overwrite:
			raise ValueError(f"File '{filepath}' already exists. Choose a different appendix, remove the existing file or use the keyword `overwrite`.")



		with h5py.File(filepath, file_mode) as f:
			
			if group_name in f:
                raise ValueError(f"Group '{group_name}' already exists in file.")



		return
'''