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
		
	return fig, ax