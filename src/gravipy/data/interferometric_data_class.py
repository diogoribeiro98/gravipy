import numpy as np
import h5py
import os
from astropy.io import fits
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

	def to_hdf5(self, filepath, mode='write'):
		"""
		Save InterferometridData class data to hdf5 file. 
		
		If ``mode=write``, tries to create a new file. If ``mode=append``, tries to append the data to an existing file.

		Args:
			filepath (str): Output file path. Should end with '.hdf5'
			mode (str, optional): Writting mode. Defaults to 'write'.

		"""
		
		#Check mode
		valid_modes = ('write', 'append', 'overwrite')

		if mode=='write':
			if os.path.exists(filepath):
				raise FileExistsError(f"File {filepath} already exists. Consider a different name or use `mode='overwrite'.")
			
			hdf5_mode = 'w-' # Create file, fail if exists
		elif mode=='append':
			hdf5_mode = 'r+' # Read/write, file must exist
		elif mode=='overwrite':
			if os.path.exists(filepath):
				os.remove(filepath)
			hdf5_mode = 'w-' # Read/write, file must exist
		else:
			raise ValueError(f"mode must be one of {valid_modes}")

		with h5py.File(filepath, hdf5_mode) as f:

			grp = f.create_group("data")

			#Header information
			grp.attrs['filename'] 	 = self.filename
			grp.attrs['fits_header'] = self.header.tostring(sep='\n').encode('utf-8')
			
			#Obsevation information
			grp.attrs['date_obs'] 		= self.date_obs
			grp.attrs['polmode'] 		= self.polmode
			grp.attrs['resolution'] 	= self.resolution

			#Observation pointing
			grp.attrs['object'  ] 		= self.object
			grp.attrs['ra'  	] 		= self.ra
			grp.attrs['dec'  	] 		= self.dec
			
			grp.attrs['sobj'  ] 		= self.sobj
			grp.attrs['sobj_x'] 		= self.sobj_x
			grp.attrs['sobj_y'] 		= self.sobj_y
			
			grp.attrs['sobj_offx'] = self.sobj_offx
			grp.attrs['sobj_offy'] = self.sobj_offy

			#Interferometric data
			grp = f.create_group("data/array")
			grp.create_dataset('Bu' 	, data=self.Bu)
			grp.create_dataset('Bv' 	, data=self.Bv)
			
			grp.create_dataset('telescopes' , data= self.telescopes.astype('S'))
			grp.create_dataset('telpos' 	, data=self.tel_pos)
			
			#Note: To save the strings to hdf one needs to convert them to byte strings
			grp.create_dataset('bl_telescopes' , data= [arr.astype('S') for arr in self.bl_telescopes])
			grp.create_dataset('t3_telescopes' , data= [arr.astype('S') for arr in self.t3_telescopes])

			grp.create_dataset('bl_labels' , data= self.bl_labels.astype('S'))
			grp.create_dataset('t3_labels' , data= self.t3_labels.astype('S'))
	
			grp = f.create_group("data/visibility/wave")
			grp.create_dataset('polarization' 	, data= self.pol)
			grp.create_dataset('wave' 			, data= self.wave)
			grp.create_dataset('band' 			, data= self.band)
			
			grp = f.create_group("data/visibility/flux")
			grp.create_dataset('flux'			, data= self.flux)
			grp.create_dataset('flux_err'		, data= self.flux_err)
			
			grp = f.create_group("data/visibility/vis")
			grp.create_dataset('visamp' 	, data= self.visamp)
			grp.create_dataset('visamp_err' , data= self.visamp_err)			
			grp.create_dataset('visphi' 	, data= self.visphi)
			grp.create_dataset('visphi_err' , data= self.visphi_err)
			grp.create_dataset('vis_flag' 	, data= self.vis_flag)
			
			grp = f.create_group("data/visibility/vis2")
			grp.create_dataset('vis2' 		, data= self.vis2)
			grp.create_dataset('vis2_err' 	, data= self.vis2_err)			
			grp.create_dataset('vis2_flag' 	, data= self.vis2_flag)
			
			grp = f.create_group("data/visibility/t3")
			grp.create_dataset('t3amp' 		, data= self.t3amp)
			grp.create_dataset('t3amp_err' 	, data= self.t3amp_err)			
			grp.create_dataset('t3phi' 		, data= self.t3phi)
			grp.create_dataset('t3phi_err' 	, data= self.t3phi_err)			
			grp.create_dataset('t3_flag' 	, data= self.t3flag)
			
			grp = f.create_group("data/visibility/sf")
			grp.create_dataset('spatial_frequency', data= self.spatial_frequency_as)
			grp.create_dataset('spatial_frequency_t3', data= self.spatial_frequency_as_T3)

		return
	
	@staticmethod
	def from_hdf5(filepath):
		"""
		Load InterferometricData from an HDF5 file.

		Args:
			filepath (str): Path to the HDF5 file.
		Returns:
			InterferometricData: Reconstructed data class.
		"""
		with h5py.File(filepath, 'r') as f:
			# --- General info from attributes ---
			grp = f['data']
			filename = grp.attrs['filename']
			
			header = fits.Header.fromstring(grp.attrs['fits_header'], sep='\n')

			date_obs = grp.attrs['date_obs']
			polmode = grp.attrs['polmode']
			resolution = grp.attrs['resolution']
			object_ = grp.attrs['object']
			ra = grp.attrs['ra']
			dec = grp.attrs['dec']
			sobj = grp.attrs['sobj']
			sobj_x = grp.attrs['sobj_x']
			sobj_y = grp.attrs['sobj_y']
			sobj_offx = grp.attrs['sobj_offx']
			sobj_offy = grp.attrs['sobj_offy']

			# --- Array data ---
			arr = f['data/array']
			Bu = arr['Bu'][()]
			Bv = arr['Bv'][()]
			telescopes = arr['telescopes'][()].astype(str)
			tel_pos = arr['telpos'][()]
			bl_telescopes = np.array([a.astype(str) for a in arr['bl_telescopes'][()]])
			t3_telescopes = np.array([a.astype(str) for a in arr['t3_telescopes'][()]])
			bl_labels = arr['bl_labels'][()].astype(str)
			t3_labels = arr['t3_labels'][()].astype(str)

			# --- Visibility data ---
			vis_wave = f['data/visibility/wave']			
			pol  = vis_wave['polarization'][()].decode()
			wave = vis_wave['wave'][()]
			band = vis_wave['band'][()]

			vis_flux = f['data/visibility/flux']
			flux = vis_flux['flux'][()]
			flux_err = vis_flux['flux_err'][()]

			vis = f['data/visibility/vis']
			visamp = vis['visamp'][()]
			visamp_err = vis['visamp_err'][()]
			visphi = vis['visphi'][()]
			visphi_err = vis['visphi_err'][()]
			vis_flag = vis['vis_flag'][()]

			vis2 = f['data/visibility/vis2']
			vis2_data = vis2['vis2'][()]
			vis2_err = vis2['vis2_err'][()]
			vis2_flag = vis2['vis2_flag'][()]

			t3 = f['data/visibility/t3']
			t3amp = t3['t3amp'][()]
			t3amp_err = t3['t3amp_err'][()]
			t3phi = t3['t3phi'][()]
			t3phi_err = t3['t3phi_err'][()]
			t3flag = t3['t3_flag'][()]

			sf = f['data/visibility/sf']
			spatial_frequency_as = sf['spatial_frequency'][()]
			spatial_frequency_as_T3 = sf['spatial_frequency_t3'][()]

		return InterferometricData(
			filename=filename,
			header=header,
			date_obs=date_obs,
			polmode=polmode,
			resolution=resolution,
			object=object_,
			ra=ra,
			dec=dec,
			sobj=sobj,
			sobj_x=sobj_x,
			sobj_y=sobj_y,
			sobj_offx=sobj_offx,
			sobj_offy=sobj_offy,
			pol=pol,
			Bu=Bu,
			Bv=Bv,
			telescopes=telescopes,
			tel_pos=tel_pos,
			bl_telescopes=bl_telescopes,
			t3_telescopes=t3_telescopes,
			bl_labels=bl_labels,
			t3_labels=t3_labels,
			wave=wave,
			band=band,
			flux=flux,
			flux_err=flux_err,
			visamp=visamp,
			visamp_err=visamp_err,
			visphi=visphi,
			visphi_err=visphi_err,
			vis_flag=vis_flag,
			vis2=vis2_data,
			vis2_err=vis2_err,
			vis2_flag=vis2_flag,
			t3amp=t3amp,
			t3amp_err=t3amp_err,
			t3phi=t3phi,
			t3phi_err=t3phi_err,
			t3flag=t3flag,
			spatial_frequency_as=spatial_frequency_as,
			spatial_frequency_as_T3=spatial_frequency_as_T3
		)