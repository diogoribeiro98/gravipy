import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from astropy.io import fits
from pkg_resources import resource_filename

try:
    from generalFunctions import *
    set_style('show')
except (NameError, ModuleNotFoundError):
    pass

###############
# Small functions

def powerLaw(nu, p, A):
    return A * nu**p

def sqrt_fit(x,a,b):
    return a/sp.sqrt(x)+b

class GravModel():
    def __init__(self, mag, ps=1, dit=10, tel='UT', res='LOW',
                 pol=True, onaxis=False):
        self.magnitude = mag
        self.powerspec = ps
        self.dit = dit
        self.tel = tel
        self.resolution = res
        self.polarization = pol
        self.onaxis = onaxis
        if self.tel == 'UT':
            self.transmission = 0.28
            #tel_router = 8
            #tel_rinner = 1.116
            self.strehlerror = 0.4
            self.collarea = 49.29
        elif tel == 'AT':
            self.transmission = 0.18
            #tel_router = 1.80
            #tel_rinner = 0.138
            self.strehlerror = 0.05
            self.collarea = 2.53
        else:
            raise Exception('tel has to be UT or AT')      
        
        self.qe = 0.82 # e-/photon
        self.specphotons = 4.53e15  # photons/(s m^2 m)
        self.specwidth = 1e-10 #spacing of 0.1nm
        self.fibererror = 0.8
        
        self.signal = None
        self.ditnoise = None
        
        if self.resolution == 'LOW':
            self.wl = np.array([2e-06, 2.037e-06, 2.074e-06, 2.111e-06, 
                                2.148e-06, 2.185e-06, 2.222e-06, 2.259e-06, 
                                2.296e-06, 2.333e-06, 2.37e-06, 2.407e-06, 
                                2.444e-06, 2.481e-06])
            
        elif self.resolution == 'MED':
            filterlistfile = resource_filename('gravipy', 
                                               'modeldata/wavelength_medium')
            filterlist = np.genfromtxt(filterlistfile)
            filterpos = np.round(filterlist[:,0],4)*1e-6
            self.wl = filterpos
            
            
    def _getFlat(self):
        ###############    
        # Atmosphere (only for x_scale)
        atmospherefile = resource_filename('gravipy', 'modeldata/atmosphere')
        atmtrans = np.genfromtxt(atmospherefile)
        steps = int(round((atmtrans[-1,0]-atmtrans[0,0])*10000))
        start = int(round(atmtrans[0,0]*10000))
        #x_scale = np.arange(start,start+steps)/10000
        x_scale = np.arange(atmtrans[0,0]*1e-6, atmtrans[-1,0]*1e-6, 
                            self.specwidth)
                
        ###############
        # Bandpass
        if self.resolution == 'LOW':
            num_channels = 14
            bandpassfile = resource_filename('gravipy', 
                                             'modeldata/LOWRES_bandpass_P1.fits')
            bandpass = fits.open(bandpassfile)
            real_wavelength = np.zeros(2048)
            filterlist = np.zeros((num_channels,2048))
            for idx in range(2048):
                real_wavelength[idx] = bandpass[1].data[idx][4]
                filterlist[:,idx] = np.mean(bandpass[1].data[idx][2],0)
            
            # rescale to nm scale & cut wings
            
            filter_i = np.zeros((len(filterlist),len(x_scale)))
            for i in range(len(filterlist)):
                filter_i[i] = sp.interpolate.interp1d(real_wavelength[93:-1395], 
                                                    filterlist[i][93:-1395], bounds_error=False,
                                                    fill_value=0, kind='linear')(x_scale)

            # Use eff. wavbelength and band:
            wl = np.array([2e-06, 2.037e-06, 2.074e-06, 2.111e-06, 2.148e-06, 2.185e-06, 
                        2.222e-06, 2.259e-06, 2.296e-06, 2.333e-06, 2.37e-06, 2.407e-06, 
                        2.444e-06, 2.481e-06])
            wl_band = np.array([7.046e-08, 1.243e-07, 1.342e-07, 1.278e-07, 1.255e-07, 1.345e-07, 
                                1.423e-07, 1.332e-07, 1.235e-07, 1.165e-07, 1.158e-07, 1.157e-07,
                                1.12e-07, 9.597e-08])
            filter_wl = np.zeros((len(wl),len(x_scale)))
            used_filter = np.zeros((len(wl),len(x_scale)))
            for idx in range(len(wl)):
                if idx == 0:
                    start = (wl[idx] - wl_band[idx]/2/3)
                else:
                    start = (wl[idx] - (wl[idx] - wl[idx-1])/2)
                if idx == len(wl)-1:
                    stop = (wl[idx] + wl_band[idx]/2/3)
                else:
                    stop = (wl[idx] + (wl[idx+1] - wl[idx])/2)
                start_pix = find_nearest(x_scale,start)
                stop_pix = find_nearest(x_scale,stop)
                filter_wl[idx,start_pix:stop_pix]=1
                used_filter[idx] = filter_wl[idx]*filter_i[idx]
    
        elif self.resolution == 'MED':
            num_channels = 210
            filterlistfile = resource_filename('gravipy', 
                                               'modeldata/wavelength_medium')
            filterlist = np.genfromtxt(filterlistfile)
            filterpos = np.round(filterlist[:,0],4)*1e-6
            wl = filterpos

            used_filter = np.zeros((num_channels, len(x_scale)))
            for idx in range(num_channels):
                filtervalue = filterpos[idx]
                val = find_nearest(x_scale, filtervalue)
                used_filter[idx, val-11:val+12] = 1
        else:
            raise Exception('Resolution has to be LOW or MED')    
        self.wl = wl            
        
        ###############
        # BC throughput
        if self.resolution == 'LOW':
            # from monochromator measurement
            throughputfile = resource_filename('gravipy', 'modeldata/throughput')
            bc_throughput = np.genfromtxt(throughputfile)[:-1,1]
            bc_throughput_wl = np.genfromtxt(throughputfile)[:-1,0]

            # Modified TP
            throughputfile = resource_filename('gravipy', 'modeldata/throughput_mod')
            bc_throughput = np.genfromtxt(throughputfile)[:-1,1]
            bc_throughput = np.insert(bc_throughput,0,0)
            bc_throughput_wl = np.insert(bc_throughput_wl,0,x_scale[0])

            # rescale to nm scale
            bc_throughput_i = sp.interpolate.interp1d(bc_throughput_wl*1e-6, bc_throughput,
                                                        kind='linear')(x_scale)

        elif self.resolution == 'MED':
            throughputfile = resource_filename('gravipy', 'modeldata/throughput_med')
            bc_throughput_wl = np.genfromtxt(throughputfile)[:,0][:245]
            bc_throughput = np.genfromtxt(throughputfile)[:,1][:245]

            bc_throughput_i = sp.interpolate.interp1d(bc_throughput_wl*1e-6, bc_throughput, 
                                                    bounds_error=False, fill_value=0)(x_scale)
    
        flat = np.zeros(num_channels)
        for i in range(num_channels):
            flat[i] = sp.integrate.simps(used_filter[i]*bc_throughput_i)
            
        flat /= np.max(flat)
        self.flat = flat
        return wl, flat
    
    def getSignal(self, cutpos=200, plot=False, flatFlux=False, fiberCoupling=1, ndit=1):
        ### Flux Calculations
        # flux of Vega:
        # 4.53e9 photons/(s m^2 mum) = 4.53e15 photons/(s m^2 m) (Value from Cox 1997)
        self.objflux = 10**(-0.4*self.magnitude)*self.specphotons*self.specwidth
        self.recflux = self.objflux*self.collarea # photons/sec*nm
        self.recelectrons = self.recflux*self.qe*self.dit # electrons/0.1nm
    
        ###############    
        # Atmosphere
        atmospherefile = resource_filename('gravipy', 'modeldata/atmosphere')
        atmtrans = np.genfromtxt(atmospherefile)
        steps = int(round((atmtrans[-1,0]-atmtrans[0,0])*10000))
        start = int(round(atmtrans[0,0]*10000))
        #x_scale = np.arange(start,start+steps)/10000
        x_scale = np.arange(atmtrans[0,0]*1e-6, atmtrans[-1,0]*1e-6, self.specwidth)
        atmtrans_i = sp.interpolate.interp1d(atmtrans[:,0]*1e-6, atmtrans[:,1], 
                                            kind='linear')(x_scale)
        
        ################
        # Input power spectrum
        signal = np.ones(len(x_scale))*self.recelectrons
        signal_pl = powerLaw(x_scale, self.powerspec, 1)
        signal_pl = signal_pl / sp.integrate.simps(signal_pl) * sp.integrate.simps(signal)
        signal = signal_pl
    
        ###############
        # Bandpass
        if self.resolution == 'LOW':
            num_channels = 14
            bandpassfile = resource_filename('gravipy', 'modeldata/LOWRES_bandpass_P1.fits')
            bandpass = fits.open(bandpassfile)
            real_wavelength = np.zeros(2048)
            filterlist = np.zeros((num_channels,2048))
            for idx in range(2048):
                real_wavelength[idx] = bandpass[1].data[idx][4]
                filterlist[:,idx] = np.mean(bandpass[1].data[idx][2],0)
            
            # rescale to nm scale & cut wings
            
            filter_i = np.zeros((len(filterlist),len(x_scale)))
            for i in range(len(filterlist)):
                filter_i[i] = sp.interpolate.interp1d(real_wavelength[93:-1395], 
                                                    filterlist[i][93:-1395], bounds_error=False,
                                                    fill_value=0, kind='linear')(x_scale)

            # Use eff. wavbelength and band:
            wl = np.array([2e-06, 2.037e-06, 2.074e-06, 2.111e-06, 2.148e-06, 2.185e-06, 
                        2.222e-06, 2.259e-06, 2.296e-06, 2.333e-06, 2.37e-06, 2.407e-06, 
                        2.444e-06, 2.481e-06])
            wl_band = np.array([7.046e-08, 1.243e-07, 1.342e-07, 1.278e-07, 1.255e-07, 1.345e-07, 
                                1.423e-07, 1.332e-07, 1.235e-07, 1.165e-07, 1.158e-07, 1.157e-07,
                                1.12e-07, 9.597e-08])
            filter_wl = np.zeros((len(wl),len(x_scale)))
            used_filter = np.zeros((len(wl),len(x_scale)))
            for idx in range(len(wl)):
                if idx == 0:
                    start = (wl[idx] - wl_band[idx]/2/3)
                else:
                    start = (wl[idx] - (wl[idx] - wl[idx-1])/2)
                if idx == len(wl)-1:
                    stop = (wl[idx] + wl_band[idx]/2/3)
                else:
                    stop = (wl[idx] + (wl[idx+1] - wl[idx])/2)
                start_pix = find_nearest(x_scale,start)
                stop_pix = find_nearest(x_scale,stop)
                filter_wl[idx,start_pix:stop_pix]=1
                used_filter[idx] = filter_wl[idx]*filter_i[idx]
    
        elif self.resolution == 'MED':
            num_channels = 210
            filterlistfile = resource_filename('gravipy', 'modeldata/wavelength_medium')
            filterlist = np.genfromtxt(filterlistfile)
            filterpos = np.round(filterlist[:,0],4)*1e-6
            wl = filterpos

            used_filter = np.zeros((num_channels, len(x_scale)))
            for idx in range(num_channels):
                filtervalue = filterpos[idx]
                val = find_nearest(x_scale, filtervalue)
                used_filter[idx, val-11:val+12] = 1
        else:
            raise Exception('Resolution has to be LOW or MED')    
        self.wl = wl

        
        ###############
        # BC throughput
        if self.resolution == 'LOW':
            # from monochromator measurement
            throughputfile = resource_filename('gravipy', 'modeldata/throughput')
            bc_throughput = np.genfromtxt(throughputfile)[:-1,1]
            bc_throughput_wl = np.genfromtxt(throughputfile)[:-1,0]

            # Modified TP
            throughputfile = resource_filename('gravipy', 'modeldata/throughput_mod')
            bc_throughput = np.genfromtxt(throughputfile)[:-1,1]
            bc_throughput = np.insert(bc_throughput,0,0)
            bc_throughput_wl = np.insert(bc_throughput_wl,0,x_scale[0])

            # rescale to nm scale
            bc_throughput_i = sp.interpolate.interp1d(bc_throughput_wl*1e-6, bc_throughput,
                                                        kind='linear')(x_scale)

        elif self.resolution == 'MED':
            throughputfile = resource_filename('gravipy', 'modeldata/throughput_med')
            bc_throughput_wl = np.genfromtxt(throughputfile)[:,0][:245]
            bc_throughput = np.genfromtxt(throughputfile)[:,1][:245]

            bc_throughput_i = sp.interpolate.interp1d(bc_throughput_wl*1e-6, bc_throughput, 
                                                    bounds_error=False, fill_value=0)(x_scale)

        ##############
        # Combined transmission:
        if plot:
            for i in range(len(wl)):
                plt.plot(x_scale,used_filter[i]*atmtrans_i*bc_throughput_i*self.transmission)
            hide_spines(outwards=False)
            #plt.plot(x_scale, atm_trans_i*np.max(used_filter[i]*atmtrans_i*bc_throughput_i*self.transmission), color='k', lw=2)
            plt.show()    
        
        det_signal = np.zeros(num_channels)
        for i in range(num_channels):
            det_signal[i] = sp.integrate.simps(used_filter[i]*atmtrans_i*
                                               signal*bc_throughput_i)*self.transmission
        
        if flatFlux:
            self._getFlat()
            det_signal /= self.flat

        det_signal_final = det_signal*self.strehlerror*self.fibererror*fiberCoupling
        if self.polarization:
            det_signal_final /= 2
        if self.onaxis:
            det_signal_final /= 2
            
        det_signal_final *= ndit
        self.signal = det_signal_final
        
        if plot:
            plt.figure(figsize=(8,3))
            plt.plot(wl*1e6,det_signal_final,marker='o')
            plt.xlabel('Wavelength [$\mu$m]')
            plt.ylabel('electrons per dit')
            hide_spines()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.show()

        return wl, det_signal_final
    
    #################
    ## Noise & SNR
    def getDITNoise(self, extSignal=None, ro=None):
        if self.resolution == 'LOW':
            noisefile = resource_filename('gravipy', 'modeldata/background_noise_LOW')
            background = np.genfromtxt(noisefile)
            #ro_a = 14.683
            #ro_b = 0.389
            ro_a = 12.653 
            ro_b = 4.97
        elif self.resolution == 'MED':
            noisefile = resource_filename('gravipy', 'modeldata/background_noise_MED')
            background = np.genfromtxt(noisefile)
            ro_a = 12.653 
            ro_b = 4.97
        else:
            raise Exception('Resolution must be LOW or MED')
        if ro is not None:
            self.readout_noise = ro
        else:
            self.readout_noise = sqrt_fit(self.dit, ro_a, ro_b)
        if extSignal is None:
            print('Use Signal from Class')
            if self.signal is None:
                self.getSignal() 
            extSignal = self.signal
        elif extSignal is 0:
            print('No signal used for Noise')
            extSignal = np.zeros_like(np.array(background))

        self.backgroundNoise = np.array(background)*self.dit
        
        ditnoise = np.sqrt(extSignal + self.backgroundNoise + self.readout_noise**2)
        self.ditnoise = ditnoise
        return ditnoise
    
    
    def getDITNoiseNEW(self, extSignal=None):
        """
        New version assumes a constant readout noise
        detector seems to be always read with fowler 16, should NOT depend on dit!
        """
        if self.resolution == 'LOW':
            noisefile = resource_filename('gravipy', 'modeldata/background_noise_LOW_new')
            background = np.genfromtxt(noisefile)
            self.readout_noise = 12.65
        elif self.resolution == 'MED':
            noisefile = resource_filename('gravipy', 'modeldata/background_noise_MED_new')
            background = np.genfromtxt(noisefile)
            self.readout_noise = 9.48
        else:
            raise Exception('Resolution must be LOW or MED')
        if extSignal is None:
            print('Use Signal from Class')
            if self.signal is None:
                self.getSignal() 
                extSignal = self.signal
            else:
                extSignal = self.signal
        elif extSignal is 0:
            print('No signal used for Noise')
            extSignal = np.zeros_like(np.array(background))
            
        self.backgroundNoise = np.array(background)*self.dit
        
        ditnoise = np.sqrt(extSignal + self.backgroundNoise + self.readout_noise**2)
        self.ditnoise = ditnoise
        return ditnoise
    
    def getExpNoise(self, exp=308, overhead=0.263, extSignal=None):
        print('Noise for Dit=%i, Exp=%i, OHs=%.3f' % (self.dit, exp, overhead))
        if extSignal is not None and extSignal is not 0:
            print('Use given signal. Careful: Should be per DIT, not per EXP!')
        self.getDITNoise(extSignal=extSignal)    
        if exp == 0:
            print('1 exp, no prefactor')
            self.expnoise = self.ditnoise
            return self.ditnoise
        else:
            nDIT = (exp/(self.dit+overhead))
            print('%i NDIT' % nDIT)
            if nDIT < 1:
                raise ValueError('DIT longer than exposure')
            pref = np.sqrt(nDIT)
            self.expnoise = pref*self.ditnoise
            return self.expnoise
        
    
    def getExpSNR(self, exp=308, overhead=0.263, extSignal=None):
        print('SNR for Dit=%i, Exp=%i, OHs=%.3f' % (self.dit, exp, overhead))
        if extSignal is not None and extSignal is not 0:
            print('Use given signal. Careful: Should be per DIT, not per EXP!')
        if self.ditnoise is None:
            self.getDITNoise(extSignal=extSignal)    
        if extSignal is None:
            self.getSignal()    
            extSignal = self.signal
        if exp == 0:
            print('1 exp, no prefactor')
            self.SNR = extSignal/self.ditnoise
            return self.SNR
        else:
            nDIT = np.floor(exp/(self.dit+overhead))
            if nDIT < 1:
                raise ValueError('DIT longer than exposure')
            pref = np.sqrt(nDIT)
            self.SNR = pref*extSignal/self.ditnoise
            return self.SNR    
        
    def getVis2SNRPS(self, ndit=1):
        """
        Returns vis2 noise of a point source (vis2=1)
        """
        channels = len(self.wl)
        vis2PS = np.ones((6, channels))
        noise = self.getDITNoise()
        # ATTENTION do I treat the laser right?
        readout = self.readout_noise + np.sqrt(self.backgroundNoise)
        # ATTENTION is this in electrons?
        noisephotons = np.sqrt(self.getSignal()[1])
        photons = np.sqrt(self.getSignal()[1])
        nom = np.sqrt(ndit)*vis2PS*photons**2
        denom = np.sqrt(2*photons**3*vis2PS+photons**2+
                        readout**2*(2*photons**2*vis2PS+2*photons+1/4)+
                        readout**4)
        return nom/denom
        
        
    def getVis2noise(self, vis2, num_samples, photons, readout):
        nom = np.sqrt(num_samples)*vis2*photons**2
        denom = np.sqrt(2*photons**3*vis2+photons**2+
                        readout**2*(2*photons**2*vis2+2*photons+1/4)+
                        readout**4)
        return nom/denom
    
        
        
