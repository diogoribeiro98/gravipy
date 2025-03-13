import matplotlib.pyplot as plt
from .interferometric_data_class import InterferometricData
from ..tools.colors import colors_baseline, colors_closure

def plot_interferometric_data(idata):

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