from .gravifit import GraviFit
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np

def plot_corner_results(
		file_list,
		radec_window = 0.5,
		mag_window   = 0.8,
		bkg_window=2):

	#Read in files, if just one:
	if isinstance(file_list, str):  
		file_list = [file_list]
	
	nfiles = len(file_list)

	#Create a class where the data will be loaded to
	gfit = GraviFit(loglevel='WARNING')

	#Load the first file and use it as reference
	gfit.load_hdf(filename=file_list[0])
	ndim       = gfit.clean_chain.shape[1]

	#Load variable names
	labels = gfit.mcmc_variables

	#Check that all files have the same parameters
	for idx,f in enumerate(file_list):
		#Load file
		gfit.load_hdf(filename=f)		
		
		# Check mcmc variables match
		if np.any(gfit.mcmc_variables!=labels):
			raise ValueError(f'The mcmc variables in file {idx} do not match the reference one')

	#Get ordered indices to make sure ra appears before dec
	sort_indices = sorted(
		range(len(labels)), 
		key=lambda i: ( 
			labels[i] == "background_flux",
			not labels[i].endswith('_ra'),
            not labels[i].endswith('_dec'),
            not labels[i].endswith('_mag'),
            not labels[i].endswith('_alpha'),
            labels[i]))

	labels = labels[sort_indices]

	#Fetch all chains, and calculate the average as to ease display
	chain_list  = np.empty(nfiles, dtype=object)
	fitpos_list = np.empty((nfiles,ndim))
	date_list = np.empty(nfiles, dtype=object)

	for idx,f in enumerate(file_list):
	
		gfit.load_hdf(filename=f)		
	
		chain_list[idx] = gfit.clean_chain[:, sort_indices]

		#Retrieve fit values
		for jj, key in enumerate(labels):
			fitpos_list[idx,jj] = gfit.fit_parameters[key].value 

		#Get dates
		date_list[idx] = gfit.idata.date_obs

	#Calculate central values
	central_values = np.median(fitpos_list, axis=0)
	
	#
	# Corner plot
	#

	#Create corner plot
	fig, axes = plt.subplots(ndim, ndim, figsize=(1.5 * ndim, 1.5 * ndim), dpi=200)
	
	#Add Field of view plot to show positions
	fov_ax = fig.add_axes([0.55, 0.55, 0.3, 0.3])  # Overlapping figure

	#Setup corner plot according to dimensions
	for row in range(ndim):
		for col in range(ndim):

			ax = axes[row, col]

			#Remove upper triangle of plots
			if col > row:
				ax.axis('off')
				continue

			if row==col:
				ax.spines['top'].set_visible(False)
				ax.spines['left'].set_visible(False)
				ax.spines['right'].set_visible(False)
				ax.set_yticks([])
				
			#Remove ticks from remaining plots 
			if row < ndim-1:
				ax.set_xticklabels([])
				ax.set_xticks([])
			if col>0:
				ax.set_yticklabels([])
				ax.set_yticks([])
		
			#Add labels to left and bottom plots
			if row==ndim-1:
				ax.set_xlabel(labels[col])
			if col==0 and row != 0:
				ax.set_ylabel(labels[row])
	
			if row==col:

				for idx in range(nfiles):

					#Estimage histogram
					chain = chain_list[idx][:,row] 
					n, bins = np.histogram(chain, bins='fd')
					
					ax.hist(chain, bins=bins, density=False, histtype='stepfilled', alpha=0.2)
					ax.hist(chain, bins=bins, density=False, histtype='step', alpha=1)

					#Gaussian smoothing
					n = gaussian_filter(n, 3)
					x0 = (bins[:-1] + bins[1:]) / 2
					ax.plot(x0, n, color='black', ls='-.', lw=1)

					#Setup the limits
					ii = row
					if (labels[ii].startswith('source') or labels[ii].startswith('sgra'))  and (labels[ii].endswith('_ra') or labels[ii].endswith('_dec')) :
						ax.set_xlim(central_values[ii]-radec_window,central_values[ii]+radec_window)

					if labels[ii].startswith('source')  and labels[ii].endswith('_dmag') :
						ax.set_xlim(central_values[ii]-mag_window,central_values[ii]+mag_window)

					if labels[ii].startswith('sgra')  and labels[ii].endswith('_dmag') :
						ax.set_xlim(central_values[ii]-mag_window,central_values[ii]+mag_window)
			
					if labels[ii].endswith('_alpha') :
						ax.set_xlim(central_values[ii]-2,central_values[ii]+2)
			
					if labels[ii] == 'background_flux' :
						ax.set_xlim(central_values[ii]-bkg_window,central_values[ii]+bkg_window)

	return fig, ax