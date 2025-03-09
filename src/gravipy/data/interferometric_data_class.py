import numpy as np
from dataclasses import dataclass

@dataclass
class InterferometricData:

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
