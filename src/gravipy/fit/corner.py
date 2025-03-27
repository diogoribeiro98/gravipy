from .gravifit import GraviFit
import matplotlib.pyplot as plt

def plot_corner_results(file_list):

	#Read in files, if just one:
	if isinstance(file_list, str):  
		file_list = [file_list]
	
	nfiles = len(file_list)

	#Create a class where the data will be loaded to
	gfit = GraviFit(loglevel='WARNING')

	#Load the first file and use it as reference
	gfit.load_hdf(filename=file_list[0])
	ndim       = gfit.clean_chain.shape[1]

	#Load sample mcmc chain
	labels      = gfit.mcmc_variables
	chain 		= gfit.clean_chain 

	#Order the chain so that RA comes before DEC
	sort_indices = sorted(
		range(len(labels)), 
		key=lambda i: ( 
			labels[i] == "background_flux",
			not labels[i].endswith('ra'),
            not labels[i].endswith('dec'),
            not labels[i].endswith('mag'),
            not labels[i].endswith('alpha'),
            labels[i]))

	labels = labels[sort_indices]
	chain  =  chain[:, sort_indices]

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
		
			#Labels
			#Add labels to left and bottom plots
			if row==ndim-1:
				ax.set_xlabel(labels[col])
			if col==0 and row != 0:
				ax.set_ylabel(labels[row])
	


	return fig, ax