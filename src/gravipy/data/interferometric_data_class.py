import numpy as np
from dataclasses import dataclass

@dataclass
class InterferometricData:

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
	tel_pos: np.ndarray

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
