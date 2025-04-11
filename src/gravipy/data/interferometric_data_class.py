import numpy as np
from dataclasses import dataclass

@dataclass
class InterferometricData:
	"""
	A class to hold the interferometric data and associated metadata, including 
	header information, observation details, polarization, metrology corrections, 
	baseline data, telescope positions, visibility data (amplitude, phase), and more.

	Parameters:
		filename (str): The name of the data file.
		date_obs (str): The date and time of observation.
		object (str): The astronomical object being observed.
		ra (float): The right ascension of the observed object in degrees.
		dec (float): The declination of the observed object in degrees.
		sobj (float): The source object flux density.
		sobj_x (float): The source object flux density in the x-direction.
		sobj_y (float): The source object flux density in the y-direction.
		sobj_offx (float): The source object offset in the x-direction.
		sobj_offy (float): The source object offset in the y-direction.
		sobj_metrology_correction_x (dict): Metrology correction for the x-direction for the source object.
		sobj_metrology_correction_y (dict): Metrology correction for the y-direction for the source object.
		north_angle (dict): The angle of the north direction for the observation, in degrees.
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
		flux (np.ndarray): Flux per wavelength
		flux_err (np.ndarray): Flux error
	"""

	#Header information
	filename: str
	date_obs: str

	#Observation pointing
	object 	: str
	ra 		: float
	dec 	: float

	sobj   : float
	sobj_x : float
	sobj_y : float

	sobj_offx: float 
	sobj_offy: float 

	#Metrology corrections
	sobj_metrology_correction_x : dict
	sobj_metrology_correction_y : dict
	north_angle : dict
	
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

	#Flux
	flux: np.ndarray
	flux_err: np.ndarray
