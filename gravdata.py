from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import emcee
import corner
from multiprocessing import Pool
from fpdf import FPDF
from PIL import Image
from scipy import optimize 
from matplotlib import gridspec

try:
    from generalFunctions import *
    set_style('show')
except NameError:
    pass

import sys
import os 

color1 = '#C02F1D'
color2 = '#348ABD'
color3 = '#F26D21'
color4 = '#7A68A6'

from astropy.time import Time
from datetime import timedelta, datetime
def convert_date(date):
    t = Time(date)
    t2 = Time('2000-01-01T12:00:00')
    date_decimal = (t.mjd - t2.mjd)/365.25+2000
    
    date = date.replace('T', ' ')
    date = date.split('.')[0]
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return date_decimal, date


class GravData():
    def __init__(self, data, verbose=True):
        self.name = data
        self.verbose = verbose
        self.colors_baseline = np.array(['k', 'darkblue', color4, 
                                         color2, 'darkred', color1])
        self.colors_closure = np.array([color1, 'darkred', 'k', color2])
        self.baseline_labels = np.array(["UT4-3","UT4-2","UT4-1",
                                         "UT3-2","UT3-1","UT2-1"])
        self.closure_labels = np.array(["UT4-3-2","UT4-3-1","UT4-2-1","UT3-2-1"])
        
        poscatg = ['VIS_DUAL_SCI_RAW', 'VIS_SINGLE_SCI_RAW', 'VIS_SINGLE_CAL_RAW', 
           'VIS_DUAL_CAL_RAW', 'VIS_SINGLE_CALIBRATED', 'VIS_DUAL_CALIBRATED',
           'SINGLE_SCI_VIS','SINGLE_SCI_VIS_CALIBRATED','DUAL_SCI_VIS', 
           'DUAL_SCI_VIS_CALIBRATED','SINGLE_CAL_VIS','DUAL_CAL_VIS', 'ASTROREDUCED',
           'DUAL_SCI_P2VMRED']
        
        header = fits.open(self.name)[0].header
        date = header['DATE-OBS']
        if header['HIERARCH ESO INS OPTI11 ID'] == 'ONAXIS':
            onaxis = True
            print('Found onaxis data!')
        else:
            onaxis = False
            
        self.header = header
        self.date = convert_date(date)
        self.raw = False

        if 'GRAV' not in header['INSTRUME']:
            raise ValueError('File seems to be not from GRAVITY')
        else:
            datatype='RAW' # default data type RAW
        if 'HIERARCH ESO PRO TYPE' in header:
            datatype = header['HIERARCH ESO PRO TYPE']
        if 'HIERARCH ESO PRO CATG' in header:
            datacatg = header['HIERARCH ESO PRO CATG']
            if datacatg not in poscatg:
                raise ValueError('filetype is %s, which is not supported' % datacatg)
            self.datacatg = datacatg
        else:
            if self.verbose:
                print('Assume this is a raw file!')
            self.raw = True
        
        self.polmode = header['HIERARCH ESO INS POLA MODE']
        self.resolution = header['HIERARCH ESO INS SPEC RES']
        
        if not self.raw:
            if self.polmode == 'SPLIT':
                self.wlSC_P1 = fits.open(self.name)['OI_WAVELENGTH', 11].data['EFF_WAVE']*1e6
                self.wlSC_P2 = fits.open(self.name)['OI_WAVELENGTH', 12].data['EFF_WAVE']*1e6
                self.channel = len(self.wlSC_P1)
                if not datacatg == 'ASTROREDUCED':
                    self.wlFT_P1 = fits.open(self.name)['OI_WAVELENGTH', 21].data['EFF_WAVE']*1e6
                    self.wlFT_P2 = fits.open(self.name)['OI_WAVELENGTH', 22].data['EFF_WAVE']*1e6
                    
            elif self.polmode == 'COMBINED':
                self.wlSC = fits.open(self.name)['OI_WAVELENGTH', 10].data['EFF_WAVE']*1e6
                self.channel = len(self.wlSC)
                if not datacatg == 'ASTROREDUCED':
                    self.wlFT = fits.open(self.name)['OI_WAVELENGTH', 20].data['EFF_WAVE']*1e6
               
    
    def getValue(self, ext1, ext2=None, ext1num=None):
        if self.raw:
            raise ValueError('Input is a RAW file, not usable for this function')
        if ext2:
            if ext1num:
                return fits.open(self.name)[ext1, ext1num].data[ext2]
            else:
                return fits.open(self.name)[ext1].data[ext2]
        else:
            if ext1num:
                return fits.open(self.name)[ext1, ext1num].data
            else:
                return fits.open(self.name)[ext1].data

    
    def getFlux(self, mode='SC', plot=False):
        if self.raw:
            raise ValueError('Input is a RAW file, not usable for this function')
        if self.polmode == 'SPLIT':
            self.fluxtime = fits.open(self.name)['OI_FLUX', 11].data['MJD']
            if mode =='SC':
                self.fluxSC_P1 = fits.open(self.name)['OI_FLUX', 11].data['FLUX']
                self.fluxSC_P2 = fits.open(self.name)['OI_FLUX', 12].data['FLUX']
                self.fluxerrSC_P1 = fits.open(self.name)['OI_FLUX', 11].data['FLUXERR']
                self.fluxerrSC_P2 = fits.open(self.name)['OI_FLUX', 12].data['FLUXERR']
                if plot:
                    if np.ndim(self.fluxSC_P1) > 1:
                        for idx in range(len(self.fluxSC_P1)):
                            plt.errorbar(self.wlSC_P1, self.fluxSC_P1[idx], self.fluxerrSC_P1[idx], color=color1)
                            plt.errorbar(self.wlSC_P2, self.fluxSC_P2[idx], self.fluxerrSC_P2[idx], color='k')
                    else:
                        plt.errorbar(self.wlSC_P1, self.fluxSC_P1, self.fluxerrSC_P1, color=color1)
                        plt.errorbar(self.wlSC_P2, self.fluxSC_P2, self.fluxerrSC_P2, color='k')
                    plt.show()
                return self.fluxtime, self.fluxSC_P1, self.fluxerrSC_P1, self.fluxSC_P2, self.fluxerrSC_P2
            
            elif mode =='FT':
                if self.datacatg == 'ASTROREDUCED':
                    raise ValueError('Astroreduced has no FT values')
                self.fluxFT_P1 = fits.open(self.name)['OI_FLUX', 21].data['FLUX']
                self.fluxFT_P2 = fits.open(self.name)['OI_FLUX', 22].data['FLUX']
                self.fluxerrFT_P1 = fits.open(self.name)['OI_FLUX', 21].data['FLUXERR']
                self.fluxerrFT_P2 = fits.open(self.name)['OI_FLUX', 22].data['FLUXERR']
                if plot:
                    if np.ndim(self.fluxFT_P1) > 1:
                        for idx in range(len(self.fluxFT_P1)):
                            plt.errorbar(self.wlFT_P1, self.fluxFT_P1[idx], self.fluxerrFT_P1[idx], color=color1)
                            plt.errorbar(self.wlFT_P2, self.fluxFT_P2[idx], self.fluxerrFT_P2[idx], color='k')
                    else:
                        plt.errorbar(self.wlFT_P1, self.fluxFT_P1, self.fluxerrFT_P1, color=color1)
                        plt.errorbar(self.wlFT_P2, self.fluxFT_P2, self.fluxerrFT_P2, color='k')
                    plt.show()
                return self.fluxtime, self.fluxFT_P1, self.fluxerrFT_P1, self.fluxFT_P2, self.fluxerrFT_P2

            else:
                raise ValueError('Mode has to be SC or FT')
            
        elif self.polmode == 'COMBINED':
            self.fluxtime = fits.open(self.name)['OI_FLUX', 10].data['MJD']
            if mode =='SC':
                self.fluxSC = fits.open(self.name)['OI_FLUX', 10].data['FLUX']
                self.fluxerrSC = fits.open(self.name)['OI_FLUX', 10].data['FLUXERR']
                if plot:
                    if np.ndim(self.fluxSC) > 1:
                        for idx in range(len(self.fluxSC)):
                            plt.errorbar(self.wlSC, self.fluxSC[idx], self.fluxerrSC[idx])
                    else:
                        plt.errorbar(self.wlSC, self.fluxSC, self.fluxerrSC, color=color1)
                    plt.xlabel('Wavelength [$\mu$m]')
                    plt.ylabel('Flux')
                    plt.show()
                return self.fluxtime, self.fluxSC, self.fluxerrSC
            
            elif mode =='FT':
                if self.datacatg == 'ASTROREDUCED':
                    raise ValueError('Astroreduced has no FT values')
                self.fluxFT = fits.open(self.name)['OI_FLUX', 20].data['FLUX']
                self.fluxerrFT = fits.open(self.name)['OI_FLUX', 20].data['FLUXERR']
                if plot:
                    if np.ndim(self.fluxFT) > 1:
                        for idx in range(len(self.fluxFT)):
                            plt.errorbar(self.wlFT, self.fluxFT[idx], self.fluxerrFT[idx])
                    else:
                        plt.errorbar(self.wlFT, self.fluxFT, self.fluxerrFT, color=color1)
                    plt.show()
                return self.fluxtime, self.fluxFT, self.fluxerrFT



    def getIntdata(self, mode='SC', plot=False, flag=False):
        if self.raw:
            raise ValueError('Input is a RAW file, not usable for this function')
        
        fitsdata = fits.open(self.name)
        if self.polmode == 'SPLIT':
            if mode =='SC':
                self.u = fitsdata['OI_VIS', 11].data.field('UCOORD')
                self.v = fitsdata['OI_VIS', 11].data.field('VCOORD')
                spFrequ = np.sqrt(self.u**2.+self.v**2.)
                wave = self.wlSC_P1
                u_as = np.zeros((len(self.u),len(wave)))
                v_as = np.zeros((len(self.v),len(wave)))
                for i in range(0,len(self.u)):
                    u_as[i,:] = self.u[i]/(wave*1.e-6) * np.pi / 180. / 3600. # 1/as
                    v_as[i,:] = self.v[i]/(wave*1.e-6) * np.pi / 180. / 3600. # 1/as
                self.spFrequAS = np.sqrt(u_as**2.+v_as**2.)
                
                self.visampSC_P1 = fitsdata['OI_VIS', 11].data.field('VISAMP')
                self.visamperrSC_P1 = fitsdata['OI_VIS', 11].data.field('VISAMPERR')

                self.visphiSC_P1 = fitsdata['OI_VIS', 11].data.field('VISPHI')
                self.visphierrSC_P1 = fitsdata['OI_VIS', 11].data.field('VISPHIERR')
                
                self.vis2SC_P1 = fitsdata['OI_VIS2', 11].data.field('VIS2DATA')
                self.vis2errSC_P1 = fitsdata['OI_VIS2', 11].data.field('VIS2ERR')
                
                self.t3SC_P1 = fitsdata['OI_T3', 11].data.field('T3PHI')
                self.t3errSC_P1 = fitsdata['OI_T3', 11].data.field('T3PHIERR')
                self.visphiSC_P1 = fitsdata['OI_VIS', 11].data.field('VISPHI')
                self.visphierrSC_P1 = fitsdata['OI_VIS', 11].data.field('VISPHIERR')
                
                self.visampSC_P2 = fitsdata['OI_VIS', 12].data.field('VISAMP')
                self.visamperrSC_P2 = fitsdata['OI_VIS', 12].data.field('VISAMPERR')
                
                self.visphiSC_P2 = fitsdata['OI_VIS', 12].data.field('VISPHI')
                self.visphierrSC_P2 = fitsdata['OI_VIS', 12].data.field('VISPHIERR')
                
                self.vis2SC_P2 = fitsdata['OI_VIS2', 12].data.field('VIS2DATA')
                self.vis2errSC_P2 = fitsdata['OI_VIS2', 12].data.field('VIS2ERR')
                
                self.t3SC_P2 = fitsdata['OI_T3', 12].data.field('T3PHI')
                self.t3errSC_P2 = fitsdata['OI_T3', 12].data.field('T3PHIERR')
                self.visphiSC_P2 = fitsdata['OI_VIS', 12].data.field('VISPHI')
                self.visphierrSC_P2 = fitsdata['OI_VIS', 12].data.field('VISPHIERR')

                self.visampflagSC_P1 = fitsdata['OI_VIS', 11].data.field('FLAG')
                self.visampflagSC_P2 = fitsdata['OI_VIS', 12].data.field('FLAG')
                self.vis2flagSC_P1 = fitsdata['OI_VIS2', 11].data.field('FLAG')
                self.vis2flagSC_P2 = fitsdata['OI_VIS2', 12].data.field('FLAG')
                self.t3flagSC_P1 = fitsdata['OI_T3', 11].data.field('FLAG')
                self.t3flagSC_P2 = fitsdata['OI_T3', 12].data.field('FLAG')
                self.visphiflagSC_P1 = fitsdata['OI_VIS', 11].data.field('FLAG')
                self.visphiflagSC_P2 = fitsdata['OI_VIS', 12].data.field('FLAG')
                
                if flag:
                    self.visampSC_P1[self.visampflagSC_P1] = np.nan
                    self.visamperrSC_P1[self.visampflagSC_P1] = np.nan
                    self.visampSC_P2[self.visampflagSC_P2] = np.nan
                    self.visamperrSC_P2[self.visampflagSC_P2] = np.nan
                    
                    self.vis2SC_P1[self.vis2flagSC_P1] = np.nan
                    self.vis2errSC_P1[self.vis2flagSC_P1] = np.nan
                    self.vis2SC_P2[self.vis2flagSC_P2] = np.nan
                    self.vis2errSC_P2[self.vis2flagSC_P2] = np.nan
                    
                    self.t3SC_P1[self.t3flagSC_P1] = np.nan
                    self.t3errSC_P1[self.t3flagSC_P1] = np.nan
                    self.t3SC_P2[self.t3flagSC_P2] = np.nan
                    self.t3errSC_P2[self.t3flagSC_P2] = np.nan

                    self.visphiSC_P1[self.visphiflagSC_P1] = np.nan
                    self.visphierrSC_P1[self.visphiflagSC_P1] = np.nan
                    self.visphiSC_P2[self.visphiflagSC_P2] = np.nan
                    self.visphierrSC_P2[self.visphiflagSC_P2] = np.nan

                    
                
                if plot:
                    gs = gridspec.GridSpec(2,2)
                    plt.figure(figsize=(25,25))
                    axis = plt.subplot(gs[0,0])
                    for idx in range(len(self.vis2SC_P1)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visampSC_P1[idx,:], self.visamperrSC_P1[idx,:], ls='', marker='o',color=self.colors_baseline[idx%6])
                    for idx in range(len(self.vis2SC_P2)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visampSC_P2[idx,:], self.visamperrSC_P2[idx,:], ls='', marker='p',color=self.colors_baseline[idx%6])
                    plt.axhline(1, ls='--', lw=0.5)
                    plt.ylim(-0.0,1.1)
                    plt.ylabel('visibility amplitude')

                    axis = plt.subplot(gs[0,1])
                    for idx in range(len(self.vis2SC_P1)):
                        plt.errorbar(self.spFrequAS[idx,:], self.vis2SC_P1[idx,:], self.vis2errSC_P1[idx,:], ls='', marker='o',color=self.colors_baseline[idx%6])
                    for idx in range(len(self.vis2SC_P2)):
                        plt.errorbar(self.spFrequAS[idx,:], self.vis2SC_P2[idx,:], self.vis2errSC_P2[idx,:], ls='', marker='p',color=self.colors_baseline[idx%6])
                    plt.axhline(1, ls='--', lw=0.5)
                    plt.ylim(-0.0,1.1)
                    plt.ylabel('visibility squared')

                    axis = plt.subplot(gs[1,1])
                    for idx in range(len(self.vis2SC_P1)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visphiSC_P1[idx,:],self.visphierrSC_P1[idx,:], ls='', marker='o',color=self.colors_baseline[idx%6])
                    for idx in range(len(self.vis2SC_P2)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visphiSC_P2[idx,:],self.visphierrSC_P2[idx,:], ls='', marker='p',color=self.colors_baseline[idx%6])
                    plt.axhline(0, ls='--', lw=0.5)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('visibility phase')
                    
                    axis = plt.subplot(gs[1,0])
                    max_u = np.zeros((4))
                    max_u[0] = np.max(np.array([spFrequ[0],spFrequ[3],spFrequ[1]]))
                    max_u[1] = np.max(np.array([spFrequ[0],spFrequ[4],spFrequ[2]]))
                    max_u[2] = np.max(np.array([spFrequ[1],spFrequ[5],spFrequ[2]]))
                    max_u[3] = np.max(np.array([spFrequ[3],spFrequ[5],spFrequ[4]]))

                    for idx in range(len(self.t3SC_P2)):
                        max_u_as = max_u[idx%4]/(wave*1.e-6) * np.pi / 180. / 3600.
                        plt.errorbar(max_u_as, self.t3SC_P2[idx,:], self.t3errSC_P2[idx,:], marker='o',color=self.colors_closure[idx%4],linestyle='')
                    for idx in range(len(self.t3SC_P2)):
                        max_u_as = max_u[idx%4]/(wave*1.e-6) * np.pi / 180. / 3600.
                        plt.errorbar(max_u_as, self.t3SC_P1[idx,:], self.t3errSC_P1[idx,:], marker='p',color=self.colors_closure[idx%4],linestyle='')
                    plt.axhline(0, ls='--', lw=0.5)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('closure phase (deg)')
                    plt.show()
        fitsdata.close()
                
                
    def getFluxfromRAW(self, flatfile, method, skyfile=None, wavefile=None, 
                       pp_wl=None, flatflux=False):
        if not self.raw:
            raise ValueError('File has to be a RAW file for this method')
        usableMethods = ['spectrum', 'preproc', 'p2vmred', 'dualscivis']
        if method not in usableMethods:
            raise TypeError('method not available, should be one of the following: %s' % usableMethods)
            
        raw = fits.open(self.name)['IMAGING_DATA_SC'].data
        det_gain = 1.984 #e-/ADU

        if skyfile is None:
            if self.verbose:
                print('No skyfile given')
            red = raw*det_gain
        else:
            sky = fits.open(skyfile)['IMAGING_DATA_SC'].data
            red = (raw-sky)*det_gain
            
        
        if red.ndim == 3:
            tsteps = red.shape[0]

        # sum over spectra domain to find maxpos
        _speclist = np.sum(np.mean(red,0),1)
        _speclist[np.where(_speclist < 300)] = 0
        
        if self.polmode == 'SPLIT':
            numspec = 48
        elif self.polmode == 'COMBINED':
            numspec = 24
        
        if self.resolution == 'LOW':
            numchannels = 14
        elif self.resolution == 'MEDIUM':
            numchannels = 241  
        else:
            raise ValueError('High not implemented yet!')

        ## extract maxpos
        #specpos = []
        #for i in range(48):
            #if np.max(_speclist) < 100:
                #raise ValueError('Detection in Noise')
            #specpos.append(np.argmax(_speclist))
            #_speclist[np.argmax(_speclist)-5:np.argmax(_speclist)+6] = 0
        #specpos = sorted(specpos)
            
        flatfits = fits.open(flatfile)
        

        
        
        fieldstart = flatfits['PROFILE_PARAMS'].header['ESO PRO PROFILE STARTX'] - 1
        fieldstop = fieldstart + flatfits['PROFILE_PARAMS'].header['ESO PRO PROFILE NX']
        
        if flatflux:
            flatdata = flatfits['IMAGING_DATA_SC'].data[0]
            flatdata += np.min(flatdata)
            flatdata /= np.max(flatdata)
            red[:,:,fieldstart:fieldstop] = red[:,:,fieldstart:fieldstop] / flatdata

        # extract spectra with profile
        red_spectra = np.zeros((tsteps,numspec,numchannels))
        for idx in range(numspec):
            _specprofile = flatfits['PROFILE_DATA'].data['DATA%i' % (idx+1)]
            _specprofile_t = np.tile(_specprofile[0], (tsteps,1,1))
            red_spectra[:,idx] = np.sum(red[:,:,fieldstart:fieldstop]*_specprofile_t, 1)
        
        if method == 'spectrum':
            return red_spectra
        elif wavefile is None:
            raise ValueError('wavefile needed!')
        elif pp_wl is None:
            raise ValueError('pp_wl needed!')
        
        
        
        # wl interpolation
        wavefits = fits.open(wavefile)
        red_spectra_i = np.zeros((tsteps, numspec, len(pp_wl)))
        for tdx in range(tsteps):
            for idx in range(numspec):
                try:
                    red_spectra_i[tdx,idx,:] = sp.interpolate.interp1d(wavefits['WAVE_DATA_SC'].data['DATA%i' % (idx+1)][0],
                                                                    red_spectra[tdx,idx,:])(pp_wl)
                except ValueError:
                    red_spectra_i[tdx,idx,:] = sp.interpolate.interp1d(wavefits['WAVE_DATA_SC'].data['DATA%i' % (idx+1)][0],
                                                                    red_spectra[tdx,idx,:], bounds_error=False, fill_value='extrapolate')(pp_wl)
                    print('Extrapolation needed')
        
        if method == 'preproc':
            return red_spectra_i
        
        red_flux_P = np.zeros((tsteps, 4, len(pp_wl)))
        red_flux_S = np.zeros((tsteps, 4, len(pp_wl)))

        _red_spec_S = red_spectra_i[:,::2,:]
        _red_spec_P = red_spectra_i[:,1::2,:]
        _red_spec_SS = np.zeros((tsteps, 6, len(pp_wl)))
        _red_spec_PS = np.zeros((tsteps, 6, len(pp_wl)))
        for idx, i in enumerate(range(0,24,4)):
            _red_spec_SS[:, idx, :] = np.sum(_red_spec_S[:,i:i+4,:],1)
            _red_spec_PS[:, idx, :] = np.sum(_red_spec_P[:,i:i+4,:],1)

        T2BM = np.array([[0,1,0,1],
                        [1,0,0,1],
                        [1,1,0,0],
                        [0,0,1,1],
                        [0,1,1,0],
                        [1,0,1,0]])
        B2TM = np.linalg.pinv(T2BM)
        B2TM /= np.max(B2TM)
        
        for idx in range(tsteps):
            red_flux_P[idx] = np.dot(B2TM,_red_spec_PS[idx])
            red_flux_S[idx] = np.dot(B2TM,_red_spec_SS[idx])
            
        if method == 'p2vmred':
            return red_flux_P, red_flux_S
        
        if method == 'dualscivis':
            return np.sum(red_flux_P, 0), np.sum(red_flux_S, 0)
    

    #########################
    # Binary fitting
    def vis_intensity(self, s, alpha, lambda0, dlambda):
        """
        Modulated interferometric intensity
        s = B*skypos-opd1-opd2
        alphs = power law
        """
        return (lambda0/2.2)**(-1-alpha)*2*dlambda*np.sinc(2*s*dlambda/lambda0**2.)*np.exp(-2.j*np.pi*s/lambda0)
    
    
    def simulateVisdata(self, theta, constant_f=True, use_opds=False, fixedBG=True, fiberOff=None, plot=True):
        """
        Test function to see how a given theta would look like
        Theta should be a list of:
        dRA, dDEC, f1, (f2), (f3), (f4), alpha flare, (V scale), f BG, (alpha BG), 
        PC RA, PC DEC, (OPD1), (OPD2), (OPD3), (OPD4)
        
        Values in bracket are by default not used, can be activated by options:
        constant_f:     Constant coupling [True]
        use_opds:       Fit OPDs [False] 
        fixedBG:        Keep background power law [True]
        use_visscale:   Fit for a scaling in visamp [False] 
        dRA:            Guess for dRA (taken from SOFFX if not 0)
        dDEC:           Guess for dDEC (taken from SOFFY if not 0)
        
        """
        theta_names_raw = np.array(["dRA", "dDEC", "f1", "f2", "f3", "f4" , "alpha flare", "V scale", "f BG",
                                    "alpha BG", "PC RA", "PC DEC", "OPD1", "OPD2", "OPD3", "OPD4"])
        rad2as = 180 / np.pi * 3600
        try:
            if len(theta) != 16:
                print('Theta has to include the following 16 parameter:')
                print(theta_names_raw)
                raise ValueError('Wrong number of input parameter given (should be 16)')
        except(TypeError):
            print('Thetha has to include the following 16 parameter:')
            print(theta_names_raw)
            raise ValueError('Wrong number of input parameter given (should be 16)')            
        
        self.constant_f = constant_f
        self.fixedBG = fixedBG
        self.use_opds = use_opds
        self.use_visscale = False
        self.fixpos = False
        self.fixedBH = False
        
        if fiberOff is None:
            self.fiberOffX = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFX"] 
            self.fiberOffY = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFY"] 
        else:
            self.fiberOffX = fiberOff[0]
            self.fiberOffY = fiberOff[1]
        print("fiber center: %.2f, %.2f (mas)" % (self.fiberOffX, self.fiberOffY))

        nwave = self.channel
        if nwave != 14:
            raise ValueError('Only usable for 14 channels')
        self.getIntdata(plot=False, flag=False)
        u = self.u
        v = self.v
        wave = self.wlSC_P1
        R = np.zeros((6,nwave))
        R[0,:] = [28.9,16.5,15.6,16.8,17.6,16.4,15.7,17.5,18.8,20.1,20.2,20.5,22.0,28.3]
        R[1,:] = [27.2,15.8,14.9,16.0,16.7,15.7,15.1,16.4,18.2,19.6,20.0,20.3,21.7,25.3]
        R[2,:] = [28.3,16.2,15.3,16.7,17.3,16.3,15.7,17.4,18.8,20.2,20.7,21.1,22.4,27.5]
        R[3,:] = [29.1,17.0,15.9,16.6,17.1,16.6,15.8,16.9,18.8,20.5,21.0,21.3,22.0,24.4]
        R[4,:] = [28.8,16.8,16.1,16.7,17.4,16.7,16.1,17.2,19.0,20.5,21.2,21.6,22.2,25.2]
        R[5,:] = [28.0,16.0,15.0,16.2,16.6,15.7,15.3,16.4,17.8,19.3,19.8,20.0,20.8,24.4]
        dlambda = np.zeros((6,nwave))
        for i in range(0,6):
            if (nwave==11) or (nwave==14):
                dlambda[i,:] = wave/R[i,:]/2
            elif nwave==210:
                dlambda[i,:] = wave/500/2
            else:
                dlambda[i,:] = 0.03817
        
        visamp, visphi, closure = self.calc_vis(theta, u, v, wave, dlambda)
        vis2 = visamp**2
        
        if plot:
            gs = gridspec.GridSpec(2,2)
            plt.figure(figsize=(25,25))
            
            wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
            dlambda_model = np.zeros((6,len(wave_model)))
            for i in range(0,6):
                dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
            (model_visamp_full, model_visphi_full, 
            model_closure_full)  = self.calc_vis(theta, u, v, wave_model, dlambda_model)
            model_vis2_full = model_visamp_full**2.

            magu = np.sqrt(u**2.+v**2.)
            u_as = np.zeros((len(u),len(wave)))
            v_as = np.zeros((len(v),len(wave)))
            for i in range(0,len(u)):
                u_as[i,:] = u[i]/(wave*1.e-6) / rad2as
                v_as[i,:] = v[i]/(wave*1.e-6) / rad2as
            magu_as = np.sqrt(u_as**2.+v_as**2.)

            u_as_model = np.zeros((len(u),len(wave_model)))
            v_as_model = np.zeros((len(v),len(wave_model)))
            for i in range(0,len(u)):
                u_as_model[i,:] = u[i]/(wave_model*1.e-6) / rad2as
                v_as_model[i,:] = v[i]/(wave_model*1.e-6) / rad2as
            magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)

            # Visamp 
            axis = plt.subplot(gs[0,0])
            for i in range(0,6):
                plt.plot(magu_as[i,:], visamp[i,:], color=self.colors_baseline[i], ls='', marker='o')
                plt.plot(magu_as_model[i,:], model_visamp_full[i,:], color=self.colors_baseline[i])
            plt.ylabel('VisAmp')
            plt.axhline(1, ls='--', lw=0.5)
            plt.ylim(-0.0,1.1)
            #plt.xlabel('spatial frequency (1/arcsec)')
            
            # Vis2
            axis = plt.subplot(gs[0,1])
            for i in range(0,6):
                plt.errorbar(magu_as[i,:], vis2[i,:], color=self.colors_baseline[i], ls='', marker='o')
                plt.plot(magu_as_model[i,:], model_vis2_full[i,:],
                        color=self.colors_baseline[i], alpha=1.0)
            #plt.xlabel('spatial frequency (1/arcsec)')
            plt.ylabel('V2')
            plt.axhline(1, ls='--', lw=0.5)
            plt.ylim(-0.0,1.1)
            
            # T3
            axis = plt.subplot(gs[1,0])
            max_u = np.zeros((4))
            max_u[0] = np.max(np.array([magu[0],magu[3],magu[1]]))
            max_u[1] = np.max(np.array([magu[0],magu[4],magu[2]]))
            max_u[2] = np.max(np.array([magu[1],magu[5],magu[2]]))
            max_u[3] = np.max(np.array([magu[3],magu[5],magu[4]]))
            maxval = []
            for i in range(0,4):
                max_u_as = max_u[i]/(wave*1.e-6) / rad2as
                max_u_as_model = max_u[i]/(wave_model*1.e-6) / rad2as
                plt.errorbar(max_u_as, closure[i,:], color=self.colors_closure[i],ls='', marker='o')
                plt.plot(max_u_as_model, model_closure_full[i,:], color=self.colors_closure[i])
            plt.axhline(0, ls='--', lw=0.5)
            plt.xlabel('spatial frequency (1/arcsec)')
            plt.ylabel('T3Phi (deg)')
            maxval = np.max(np.abs(model_closure_full))
            if maxval < 15:
                maxplot=20
            elif maxval < 75:
                maxplot=80
            else:
                maxplot=180
            plt.ylim(-maxplot, maxplot)
            
            # VisPhi
            axis = plt.subplot(gs[1,1])
            for i in range(0,6):
                plt.errorbar(magu_as[i,:], visphi[i,:], color=self.colors_baseline[i], ls='', marker='o')
                plt.plot(magu_as_model[i,:], model_visphi_full[i,:], color=self.colors_baseline[i],alpha=1.0)
            plt.ylabel('VisPhi')
            plt.xlabel('spatial frequency (1/arcsec)')
            plt.axhline(0, ls='--', lw=0.5)
            maxval = np.max(np.abs(model_visphi_full))
            if maxval < 45:
                maxplot=50
            elif maxval < 95:
                maxplot=100
            else:
                maxplot=180
            plt.ylim(-maxplot, maxplot)
            
            plt.suptitle('dRa=%.1f, dDec=%.1f, fratio=%.1f, a=%.1f, fBG=%.1f, PCRa=%.1f, PCDec=%.1f' 
                         % (theta[0], theta[1], theta[2], theta[6], theta[8], theta[10], theta[11]), fontsize=12)
            plt.show()

        return visamp, vis2, visphi, closure


    def calc_vis(self, theta, u, v, wave, dlambda):
        mas2rad = 1e-3 / 3600 / 180 * np.pi
        rad2mas = 180 / np.pi * 3600 * 1e3
        constant_f = self.constant_f
        fixedBG = self.fixedBG
        use_opds = self.use_opds
        use_visscale = self.use_visscale
        fiberOffX = self.fiberOffX
        fiberOffY = self.fiberOffY
        fixpos = self.fixpos
        fixedBH = self.fixedBH
        specialfit = self.specialfit
        
        if fixpos:
            dRA = self.fiberOffX
            dDEC = self.fiberOffY
        else:
            dRA = theta[0]
            dDEC = theta[1]
        if constant_f:
            fluxRatio = theta[2]
        else:
            fluxRatio1 = theta[2]
            fluxRatio2 = theta[3]
            fluxRatio3 = theta[4]
            fluxRatio4 = theta[5]
        if fixedBH:
            alpha_SgrA = -0.5
        else:
            alpha_SgrA = theta[6]
        if use_visscale:
            vis_scale = theta[7]
        else:
            vis_scale = 1
        fluxRatioBG = theta[8]
        if fixedBG:
            alpha_bg = 3.
        else:
            alpha_bg = theta[9]
        phaseCenterRA = theta[10]
        phaseCenterDEC = theta[11]
        if use_opds:
            opd1 = theta[12]
            opd2 = theta[13]
            opd3 = theta[14]
            opd4 = theta[15]
            opd_bl = np.array([[opd4, opd3],
                               [opd4, opd2],
                               [opd4, opd1],
                               [opd3, opd2],
                               [opd3, opd1],
                               [opd2, opd1]])
        if specialfit:
            special_par = theta[16] 
            sp_bl = np.ones(6)*special_par
            sp_bl *= self.specialfit_bl
        alpha_S2 = 3
        
        # Flux Ratios
        if constant_f:
            f = np.ones(4)*fluxRatio
        else:
            f = np.array([fluxRatio1, fluxRatio2, fluxRatio3, fluxRatio4])
        f = 10.**f
        f_bl = np.array([[f[3],f[2]],
                         [f[3],f[1]],
                         [f[3],f[0]],
                         [f[2],f[1]],
                         [f[2],f[0]],
                         [f[1],f[0]]])

        # Calculate complex visibilities
        vis = np.zeros((6,len(wave))) + 0j
        for i in range(0,6):
            try:
                if self.fit_for[3] == 0:
                    phaseCenterRA = 0
                    phaseCenterDEC = 0
            except AttributeError:
                pass
            s_SgrA = ((phaseCenterRA)*u[i] + (phaseCenterDEC)*v[i]) * mas2rad * 1e6
            s_S2 = ((dRA+phaseCenterRA)*u[i] + (dDEC+phaseCenterDEC)*v[i]) * mas2rad * 1e6

            
            if use_opds:
                s_S2 = s_S2 + opd_bl[i,0] - opd_bl[i,1]
            if specialfit:
                s_SgrA += sp_bl[i]
                s_S2 += sp_bl[i]
    
            # u,v in 1/mas
            u_mas = u[i] / (wave*1e-6) / rad2mas
            v_mas = v[i] / (wave*1e-6) / rad2mas
            
            # interferometric intensities of all components
            intSgrA = self.vis_intensity(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
            intS2 = self.vis_intensity(s_S2, alpha_S2, wave, dlambda[i,:])
            intSgrA_center = self.vis_intensity(0, alpha_SgrA, wave, dlambda[i,:])
            intS2_center = self.vis_intensity(0, alpha_S2, wave, dlambda[i,:])
            intBG = self.vis_intensity(0, alpha_bg, wave, dlambda[i,:])
            
            vis[i,:] = ((intSgrA + 
                        np.sqrt(f_bl[i,0] * f_bl[i,1]) * intS2)/
                        (np.sqrt(intSgrA_center + f_bl[i,0] * intS2_center 
                                + fluxRatioBG * intBG) *
                        np.sqrt(intSgrA_center + f_bl[i,1] * intS2_center 
                                + fluxRatioBG * intBG)))

        
        visamp = np.abs(vis)*vis_scale
        visphi = np.angle(vis, deg=True)
        closure = np.zeros((4, len(wave)))
        closure[0,:] = visphi[0,:] + visphi[3,:] - visphi[1,:]
        closure[1,:] = visphi[0,:] + visphi[4,:] - visphi[2,:]
        closure[2,:] = visphi[1,:] + visphi[5,:] - visphi[2,:]
        closure[3,:] = visphi[3,:] + visphi[5,:] - visphi[4,:]

        visphi = visphi + 360.*(visphi<-180.) - 360.*(visphi>180.)
        closure = closure + 360.*(closure<-180.) - 360.*(closure>180.)
        return visamp, visphi, closure 
    
    
    def lnprob(self, theta, fitdata, u, v, wave, dlambda, lower, upper):
        if np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        return self.lnlike(theta, fitdata, u, v, wave, dlambda)
    
        
    def lnlike(self, theta, fitdata, u, v, wave, dlambda):       
        """
        Calculate the likelihood estimation for the MCMC run
        """
        # Model
        model_visamp, model_visphi, model_closure = self.calc_vis(theta,u,v,wave,dlambda)
        model_vis2 = model_visamp**2.
        
        #Data
        (visamp, visamp_error, visamp_flag,
         vis2, vis2_error, vis2_flag,
         closure, closure_error, closure_flag,
         visphi, visphi_error, visphi_flag) = fitdata
        
        res_visamp = np.sum(-(model_visamp-visamp)**2/visamp_error**2*(1-visamp_flag))
        res_vis2 = np.sum(-(model_vis2-vis2)**2./vis2_error**2.*(1-vis2_flag))
        res_clos = np.sum(-np.minimum((model_closure-closure)**2.,
                                      (360-(model_closure-closure))**2.)/
                          closure_error**2.*(1-closure_flag))
        res_phi = np.sum(-np.minimum((model_visphi-visphi)**2.,
                                     (360-(model_visphi-visphi))**2.)/
                          visphi_error**2.*(1-visphi_flag))
        
        ln_prob_res = 0.5 * (res_visamp * self.fit_for[0] + 
                             res_vis2 * self.fit_for[1] + 
                             res_clos * self.fit_for[2] + 
                             res_phi * self.fit_for[3])
        
        return ln_prob_res 
    
    
    def lnlike_minimize(self, theta):       
        """
        Calculate the likelihood estimation for minimizing function
        """        
        fitdata = self.fitdata
        u = self.u
        v = self.v
        wave = self.wlSC_P1
        dlambda = self.dlambda
        # Model
        model_visamp, model_visphi, model_closure = self.calc_vis(theta,u,v,wave,dlambda)
        model_vis2 = model_visamp**2.
        
        #Data
        (visamp, visamp_error, visamp_flag,
         vis2, vis2_error, vis2_flag,
         closure, closure_error, closure_flag,
         visphi, visphi_error, visphi_flag) = fitdata
        
        res_visamp = np.sum((model_visamp-visamp)**2/visamp_error**2*(1-visamp_flag))
        res_vis2 = np.sum((model_vis2-vis2)**2./vis2_error**2.*(1-vis2_flag))
        res_clos = np.sum(np.minimum((model_closure-closure)**2.,
                                      (360-(model_closure-closure))**2.)/
                          closure_error**2.*(1-closure_flag))
        res_phi = np.sum(np.minimum((model_visphi-visphi)**2.,
                                     (360-(model_visphi-visphi))**2.)/
                          visphi_error**2.*(1-visphi_flag))
        
        ln_prob_res = 0.5 * (res_visamp * self.fit_for[0] + 
                             res_vis2 * self.fit_for[1] + 
                             res_clos * self.fit_for[2] + 
                             res_phi * self.fit_for[3])
        
        return ln_prob_res     
    
    def fitBinary(self, nthreads=4, nwalkers=500, nruns=500, bestchi=True,
                  plot=True, fit_for=np.array([0.5,0.5,1.0,0.0]), constant_f=True,
                  use_opds=False, fixedBG=True, noS2=True,
                  use_visscale=False, write_results=True, flagtill=3, flagfrom=13,
                  dRA=0., dDEC=0., plotres=True, createpdf=True, bequiet=False,
                  fixpos=False, fixedBH=False, second_iteration=False,
                  dphRA=0.1, dphDec=0.1, specialpar=np.array([0,0,0,0,0,0])):
        '''s_S2
        Parameter:
        nthreads:       number of cores [4] 
        nwalkers:       number of walkers [500] 
        nruns:          number of MCMC runs [500] 
        bestchi:        Gives best chi2 (for True) or mcmc res as output [True]
        plot:           plot MCMC results [True]
        plotres:        plot fit result [True]
        write_results:  Write fit results in file [True] 
        fit_for:        weight of VA, V2, T3, VP [[0.5,0.5,1.0,0.0]] 
        constant_f:     Constant coupling [True]
        use_opds:       Fit OPDs [False] 
        noS2:           Does not do anything if OFFX and OFFY=0
        fixedBG:        Fir for background power law [False]
        use_visscale:   Fit for a scaling in visamp [False] 
        flagtill:       Flag blue channels [3] 
        flagfrom:       Flag red channels [13]
        dRA:            Guess for dRA (taken from SOFFX if not 0)
        dDEC:           Guess for dDEC (taken from SOFFY if not 0)
        bequiet:        Suppresses ALL outputs
        createpdf:      Creates a pdf with fit results and all plots [True] 
        fixpos:         Does nto fit the distance between the sources [False]
        fixedBH:        Fit for black hole power law [False]
        second_iteration: Does a second iteration with corrected error bars [False]
        '''
        self.fit_for = fit_for
        self.constant_f = constant_f
        self.use_opds = use_opds
        self.fixedBG = fixedBG
        self.use_visscale = use_visscale
        self.fixpos = fixpos
        self.fixedBH = fixedBH
        rad2as = 180 / np.pi * 3600
        if np.any(specialpar):
            self.specialfit = True
            #self.specialfit_bl = np.array([0,1,1,1,1,0])
            #self.specialfit_bl = np.array([1,0,0,0,0,1])
            #self.specialfit_bl = np.array([1,1,0,0,-1,-1])
            self.specialfit_bl = specialpar
            if not bequiet:
                print('Specialfit parameter applied to BLs:')
                nonzero = np.nonzero(self.specialfit_bl)[0]
                print(*list(nonzero*self.specialfit_bl[nonzero]))
                print('\n')
        else:
            self.specialfit = False
        specialfit = self.specialfit


        # Get data from file
        nwave = self.channel
        self.getIntdata(plot=False, flag=False)
        MJD = fits.open(self.name)[0].header["MJD-OBS"]
        u = self.u
        v = self.v
        wave = self.wlSC_P1
        self.fiberOffX = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFX"] 
        self.fiberOffY = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFY"] 
        if not bequiet:
            print("fiber center: %.2f, %.2f (mas)" % (self.fiberOffX,
                                                    self.fiberOffY))
        if dRA == 0 and dDEC == 0:
            if self.fiberOffX != 0 and self.fiberOffY != 0:
                dRA = self.fiberOffX
                dDEC = self.fiberOffY
            if self.fiberOffX == 0 and self.fiberOffY == 0:
                if noS2:
                    if not bequiet:
                        print('No Fiber offset, if you want to fit this file use noS2=False')
                    return 0
            if dRA == 0 and dDEC == 0:
                if not bequiet:
                    print('Fiber offset is zero, guess for dRA & dDEC should be given with function')
        else:
            print('Guess for RA & DEC from function as: %.2f, %.2f' % (dRA, dDEC))
        
        stname = self.name.find('GRAVI')        
        txtfilename = 'binaryfit_' + self.name[stname:-5] + '.txt'
        if write_results:
            txtfile = open(txtfilename, 'w')
            txtfile.write('# Results of binary fit for %s \n' % self.name[stname:])
            txtfile.write('# Lines are: Best chi2, MCMC result, MCMC error -, MCMC error + \n')
            txtfile.write('# Rowes are: dRA, dDEC, f1, f2, f3, f4, alpha flare, V scale, f BG, alpha BG, PC RA, PC DEC, OPD1, OPD2, OPD3, OPD4 \n')
            txtfile.write('# Parameter which are not fitted have 0.0 as error \n')
            txtfile.write('# MJD: %f \n' % MJD)
            txtfile.write('# OFFX: %f \n' % self.fiberOffX)
            txtfile.write('# OFFY: %f \n\n' % self.fiberOffY)
            if second_iteration:
                txtfilename_cor = 'binaryfit_' + self.name[stname:-5] + '_2nd.txt'
                txtfile_cor = open(txtfilename_cor, 'w')
                txtfile_cor.write('# Results of binary fit for %s from 2nd iteration\n' % self.name[stname:])
                txtfile_cor.write('# Lines are: Best chi2, MCMC result, MCMC error -, MCMC error + \n')
                txtfile_cor.write('# Rowes are: dRA, dDEC, f1, f2, f3, f4, alpha flare, V scale, f BG, alpha BG, PC RA, PC DEC, OPD1, OPD2, OPD3, OPD4 \n')
                txtfile_cor.write('# Parameter which are not fitted have 0.0 as error \n')
                txtfile_cor.write('# MJD: %f \n' % MJD)
                txtfile_cor.write('# OFFX: %f \n' % self.fiberOffX)
                txtfile_cor.write('# OFFY: %f \n\n' % self.fiberOffY)
        
        if self.polmode == 'COMBINED':
            R = np.zeros((6,nwave))
            if nwave == 11:
                R[0,:] = [32.9,20.6,20.3,19.3,19.2,16.1,18.3,20.8,21.2,21.7,23.4]
                R[1,:] = [31.8,18.6,17.5,18.5,19.8,16.8,19.8,22.7,22.6,22.8,22.7]
                R[2,:] = [31.8,19.1,19.0,18.7,18.9,16.3,19.1,21.6,22.2,22.5,23.6]
                R[3,:] = [29.9,18.3,18.6,20.6,23.5,19.5,22.7,25.8,25.4,26.8,26.2]
                R[4,:] = [30.8,18.0,17.6,19.3,22.3,19.4,23.3,26.5,26.3,27.7,24.9]
                R[5,:] = [29.7,18.1,18.1,18.1,18.6,16.5,19.6,22.4,22.8,22.8,22.3]
            elif nwave == 14:
                R[0,:] = [28.9,16.5,15.6,16.8,17.6,16.4,15.7,17.5,18.8,20.1,20.2,20.5,22.0,28.3]
                R[1,:] = [27.2,15.8,14.9,16.0,16.7,15.7,15.1,16.4,18.2,19.6,20.0,20.3,21.7,25.3]
                R[2,:] = [28.3,16.2,15.3,16.7,17.3,16.3,15.7,17.4,18.8,20.2,20.7,21.1,22.4,27.5]
                R[3,:] = [29.1,17.0,15.9,16.6,17.1,16.6,15.8,16.9,18.8,20.5,21.0,21.3,22.0,24.4]
                R[4,:] = [28.8,16.8,16.1,16.7,17.4,16.7,16.1,17.2,19.0,20.5,21.2,21.6,22.2,25.2]
                R[5,:] = [28.0,16.0,15.0,16.2,16.6,15.7,15.3,16.4,17.8,19.3,19.8,20.0,20.8,24.4]
        else:
            R = np.zeros((6,nwave))
            if nwave == 11:
                R[0,:] = [25.0,18.3,20.1,20.7,20.0,16.7,17.9,20.1,20.5,21.3,23.9]
                R[1,:] = [24.0,16.5,15.9,17.2,17.6,15.3,17.3,19.7,20.2,20.8,22.4]
                R[2,:] = [26.2,19.5,19.8,19.3,18.9,16.1,17.6,19.9,20.7,21.3,23.4]
                R[3,:] = [24.6,16.4,15.5,17.2,18.2,16.9,19.5,22.2,22.6,23.0,22.5]
                R[4,:] = [26.4,17.5,16.2,17.2,17.9,16.3,18.8,21.4,21.6,21.8,22.1]
                R[5,:] = [27.4,18.8,17.5,17.6,17.4,15.2,16.8,19.2,19.8,20.0,21.1]
            elif nwave == 14:
                R[0,:] = [28.9,16.5,15.6,16.8,17.6,16.4,15.7,17.5,18.8,20.1,20.2,20.5,22.0,28.3]
                R[1,:] = [27.2,15.8,14.9,16.0,16.7,15.7,15.1,16.4,18.2,19.6,20.0,20.3,21.7,25.3]
                R[2,:] = [28.3,16.2,15.3,16.7,17.3,16.3,15.7,17.4,18.8,20.2,20.7,21.1,22.4,27.5]
                R[3,:] = [29.1,17.0,15.9,16.6,17.1,16.6,15.8,16.9,18.8,20.5,21.0,21.3,22.0,24.4]
                R[4,:] = [28.8,16.8,16.1,16.7,17.4,16.7,16.1,17.2,19.0,20.5,21.2,21.6,22.2,25.2]
                R[5,:] = [28.0,16.0,15.0,16.2,16.6,15.7,15.3,16.4,17.8,19.3,19.8,20.0,20.8,24.4]
            
        dlambda = np.zeros((6,nwave))
        for i in range(0,6):
            if (nwave==11) or (nwave==14):
                dlambda[i,:] = wave/R[i,:]/2
            elif nwave==210:
                dlambda[i,:] = wave/500/2
            else:
                dlambda[i,:] = 0.03817
        self.dlambda = dlambda
        self.wave = wave

        # Initial guesses
        size = 4
        dRA_init = np.array([dRA,dRA-size,dRA+size])
        dDEC_init = np.array([dDEC,dDEC-size,dDEC+size])

        fr_start = np.log10(0.1)
        flux_ratio_1_init = np.array([fr_start, np.log10(0.01), np.log10(10.)])
        flux_ratio_2_init = np.array([fr_start, np.log10(0.001), np.log10(10.)])
        flux_ratio_3_init = np.array([fr_start, np.log10(0.001), np.log10(10.)])
        flux_ratio_4_init = np.array([fr_start, np.log10(0.001), np.log10(10.)])

        alpha_SgrA_init = np.array([-1.,-10.,10.])
        vis_scale_init = np.array([0.8,0.1,1.2])
        flux_ratio_bg_init = np.array([0.1,0.,20.])
        color_bg_init = np.array([3.,-10.,10.])

        size = 5
        phase_center_RA = dphRA
        phase_center_DEC = dphDec

        phase_center_RA_init = np.array([phase_center_RA,phase_center_RA-size,phase_center_RA+size])
        phase_center_DEC_init = np.array([phase_center_DEC,phase_center_DEC-size,phase_center_DEC+size])

        opd_max = 0.5 # maximum opd in microns (lambda/4)
        opd_1_init = [0.1,-opd_max,opd_max]
        opd_2_init = [0.1,-opd_max,opd_max]
        opd_3_init = [0.1,-opd_max,opd_max]
        opd_4_init = [0.1,-opd_max,opd_max]
        special_par = [-0.15, -2, 2]

        # initial fit parameters 
        theta = np.array([dRA_init[0],dDEC_init[0],flux_ratio_1_init[0],flux_ratio_2_init[0],
                          flux_ratio_3_init[0],flux_ratio_4_init[0],alpha_SgrA_init[0],vis_scale_init[0],
                          flux_ratio_bg_init[0],color_bg_init[0],phase_center_RA_init[0],
                          phase_center_DEC_init[0],opd_1_init[0],opd_2_init[0],opd_3_init[0],opd_4_init[0],special_par[0]])

        # lower limit on fit parameters 
        theta_lower = np.array([dRA_init[1],dDEC_init[1],flux_ratio_1_init[1],flux_ratio_2_init[1],
                                flux_ratio_3_init[1],flux_ratio_4_init[1],alpha_SgrA_init[1],vis_scale_init[1],
                                flux_ratio_bg_init[1],color_bg_init[1],phase_center_RA_init[1],
                                phase_center_DEC_init[1],opd_1_init[1],opd_2_init[1],opd_3_init[1],opd_4_init[1], special_par[1]])

        # upper limit on fit parameters 
        theta_upper = np.array([dRA_init[2],dDEC_init[2],flux_ratio_1_init[2],flux_ratio_2_init[2],
                                flux_ratio_3_init[2],flux_ratio_4_init[2],alpha_SgrA_init[2],
                                vis_scale_init[2],flux_ratio_bg_init[2],color_bg_init[2],phase_center_RA_init[2],
                                phase_center_DEC_init[2],opd_1_init[2],opd_2_init[2],opd_3_init[2],opd_4_init[2], special_par[2]])

        theta_names = np.array(["dRA", "dDEC", "f1", "f2", "f3", "f4" , r"$\alpha_{flare}$", r"|V| sc", r"$f_{bg}$",
                                r"$\alpha_{bg}$", r"$RA_{PC}$", r"$DEC_{PC}$", "OPD1", "OPD2", "OPD3", "OPD4", "special"])
        theta_names_raw = np.array(["dRA", "dDEC", "f1", "f2", "f3", "f4" , "alpha flare", "V scale", "f BG",
                                    "alpha BG", "PC RA", "PC DEC", "OPD1", "OPD2", "OPD3", "OPD4", "special"])


        ndim = len(theta)
        todel = []
        if fixpos:
            todel.append(0)
            todel.append(1)
        if constant_f:
            todel.append(3)
            todel.append(4)
            todel.append(5)
        if fixedBH:
            todel.append(6)
        if not use_visscale:
            todel.append(7)
        if fixedBG:
            todel.append(9)
        if fit_for[3] == 0:
            todel.append(10)
            todel.append(11)
        if not use_opds:
            todel.append(12)
            todel.append(13)
            todel.append(14)
            todel.append(15)
        if not specialfit:
            todel.append(16)
        ndof = 16 - len(todel)
        

                
        # Get data
        if self.polmode == 'SPLIT':
            visamp_P = [self.visampSC_P1, self.visampSC_P2]
            visamp_error_P = [self.visamperrSC_P1, self.visamperrSC_P2]
            visamp_flag_P = [self.visampflagSC_P1, self.visampflagSC_P2]
            
            vis2_P = [self.vis2SC_P1, self.vis2SC_P2]
            vis2_error_P = [self.vis2errSC_P1, self.vis2errSC_P2]
            vis2_flag_P = [self.vis2flagSC_P1, self.vis2flagSC_P2]

            closure_P = [self.t3SC_P1, self.t3SC_P2]
            closure_error_P = [self.t3errSC_P1, self.t3errSC_P2]
            closure_flag_P = [self.t3flagSC_P1, self.t3flagSC_P2]
            
            visphi_P = [self.visphiSC_P1, self.visphiSC_P2]
            visphi_error_P = [self.visphierrSC_P1, self.visphierrSC_P2]
            visphi_flag_P = [self.visampflagSC_P1, self.visampflagSC_P2]
            
            ndit = np.shape(self.visampSC_P1)[0]//6
            if not bequiet:
                print('NDIT = %i' % ndit)
            
            for dit in range(ndit):
                if write_results and ndit > 1:
                    txtfile.write('# DIT %i \n' % dit)
                    if second_iteration:
                        txtfile_cor.write('# DIT %i \n' % dit)
                if createpdf:
                    savetime = str(datetime.now()).replace('-', '')
                    savetime = savetime.replace(' ', '-')
                    savetime = savetime.replace(':', '')
                    self.savetime = savetime
                    if ndit == 1:
                        pdffilename = 'binaryfit_' + self.name[stname:-5] + '.pdf'
                    else:
                        pdffilename = 'binaryfit_' + self.name[stname:-5] + '_DIT' + str(dit) + '.pdf'
                    pdf = FPDF(orientation='P', unit='mm', format='A4')
                    pdf.add_page()
                    pdf.set_font("Helvetica", size=12)
                    pdf.set_margins(20,20)
                    if ndit == 1:
                        pdf.cell(0, 10, txt="Fit report for %s" % self.name[stname:], ln=2, align="C", border='B')
                    else:
                        pdf.cell(0, 10, txt="Fit report for %s, dit %i" % (self.name[stname:], dit), ln=2, align="C", border='B')
                    pdf.ln()
                    pdf.cell(40, 6, txt="Fringe Tracker", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=self.header["ESO FT ROBJ NAME"], ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Science Object", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=self.header["ESO INS SOBJ NAME"], ln=1, align="L", border=0)
                    pdf.cell(40, 6, txt="Science Offset X", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(self.header["ESO INS SOBJ OFFX"]), 
                            ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Science Offset Y", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(self.header["ESO INS SOBJ OFFY"]), 
                            ln=1, align="L", border=0)
                    pdf.cell(40, 6, txt="Fit for Visamp", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(fit_for[0]), ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Fit for Vis2", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(fit_for[1]), ln=1, align="L", border=0)
                    pdf.cell(40, 6, txt="Fit for cl. Phase", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(fit_for[2]), ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Fit for Visphi", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(fit_for[3]), ln=1, align="L", border=0)
                    pdf.cell(40, 6, txt="Constant coulping", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(constant_f), ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Fixed Bg", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(fixedBG), ln=1, align="L", border=0)
                    pdf.cell(40, 6, txt="Scale Visamp", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(use_visscale), ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Flag before/after", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(flagtill) + '/' + str(flagfrom), 
                            ln=1, align="L", border=0)
                    pdf.cell(40, 6, txt="Result: Best Chi2", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(bestchi), ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Fit OPDs", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(use_opds), ln=1, align="L", border=0)
                    pdf.cell(40, 6, txt="Fixpos", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(fixpos), ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Fixed BH", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(fixedBH), ln=1, align="L", border=0)
                    pdf.ln()
                
                if not bequiet:
                    print('Run MCMC for DIT %i' % (dit+1))
                ditstart = dit*6
                ditstop = ditstart + 6
                t3ditstart = dit*4
                t3ditstop = t3ditstart + 4
                
                for idx in range(2):
                    visamp = visamp_P[idx][ditstart:ditstop]
                    visamp_error = visamp_error_P[idx][ditstart:ditstop]
                    visamp_flag = visamp_flag_P[idx][ditstart:ditstop]
                    vis2 = vis2_P[idx][ditstart:ditstop]
                    vis2_error = vis2_error_P[idx][ditstart:ditstop]
                    vis2_flag = vis2_flag_P[idx][ditstart:ditstop]
                    closure = closure_P[idx][t3ditstart:t3ditstop]
                    closure_error = closure_error_P[idx][t3ditstart:t3ditstop]
                    closure_flag = closure_flag_P[idx][t3ditstart:t3ditstop]
                    visphi = visphi_P[idx][ditstart:ditstop]
                    visphi_error = visphi_error_P[idx][ditstart:ditstop]
                    visphi_flag = visphi_flag_P[idx][ditstart:ditstop]

                    # further flag if visamp/vis2 if >1 or NaN, and replace NaN with 0 
                    with np.errstate(invalid='ignore'):
                        visamp_flag1 = (visamp > 1) | (visamp < 1.e-5)
                    visamp_flag2 = np.isnan(visamp)
                    visamp_flag_final = ((visamp_flag) | (visamp_flag1) | (visamp_flag2))
                    visamp_flag = visamp_flag_final
                    visamp = np.nan_to_num(visamp)
                    visamp_error[visamp_flag] = 1.
                
                    with np.errstate(invalid='ignore'):
                        vis2_flag1 = (vis2 > 1) | (vis2 < 1.e-5) 
                    vis2_flag2 = np.isnan(vis2)
                    vis2_flag_final = ((vis2_flag) | (vis2_flag1) | (vis2_flag2))
                    vis2_flag = vis2_flag_final
                    vis2 = np.nan_to_num(vis2)
                    vis2_error[vis2_flag] = 1.

                    if ((flagtill > 0) and (flagfrom > 0)):
                        p = flagtill
                        t = flagfrom
                        if idx == 0 and dit == 0:
                            if not bequiet:
                                print('using channels from #%i to #%i' % (p, t))
                        visamp_flag[:,0:p] = True
                        vis2_flag[:,0:p] = True
                        visphi_flag[:,0:p] = True
                        closure_flag[:,0:p] = True

                        visamp_flag[:,t] = True
                        vis2_flag[:,t] = True
                        visphi_flag[:,t] = True
                        closure_flag[:,t] = True
                    
                    width = 1e-1
                    pos = np.ones((nwalkers,ndim))
                    for par in range(ndim):
                        if par in todel:
                            pos[:,par] = theta[par]
                        else:
                            pos[:,par] = theta[par] + width*np.random.randn(nwalkers)
                    if not bequiet:
                        print('Run MCMC for Pol %i' % (idx+1))
                    fitdata = [visamp, visamp_error, visamp_flag,
                            vis2, vis2_error, vis2_flag,
                            closure, closure_error, closure_flag,
                            visphi, visphi_error, visphi_flag]

                    if nthreads == 1:
                        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, 
                                                            args=(fitdata, u, v, wave,
                                                                dlambda, theta_lower,
                                                                theta_upper))
                        if bequiet:
                            sampler.run_mcmc(pos, nruns, progress=False)
                        else:
                            sampler.run_mcmc(pos, nruns, progress=True)
                    else:
                        with Pool(processes=nthreads) as pool:
                            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, 
                                                            args=(fitdata, u, v, wave,
                                                                dlambda, theta_lower,
                                                                theta_upper),
                                                            pool=pool)
                            if bequiet:
                                sampler.run_mcmc(pos, nruns, progress=False) 
                            else:
                                sampler.run_mcmc(pos, nruns, progress=True)     
                            
                    if not bequiet:
                        print("---------------------------------------")
                        print("Mean acceptance fraction: %.2f"  % np.mean(sampler.acceptance_fraction))
                        print("---------------------------------------")
                    if createpdf:
                        pdf.cell(0, 10, txt="Polarization  %i" % (idx+1), ln=2, align="C", border='B')
                        pdf.cell(0, 10, txt="Mean acceptance fraction: %.2f"  %
                                np.mean(sampler.acceptance_fraction), 
                                ln=2, align="L", border=0)
                    samples = sampler.chain
                    mostprop = sampler.flatchain[np.argmax(sampler.flatlnprobability)]

                    # for debuggin: save the chain
                    #samples = np.load("samples_MCMC_groupedM_%i.npy" % idx)
                    #mostprop = np.load("samples_MCMC_groupedM_mostprop_%i.npy" % idx)
                    
                    clsamples = np.delete(samples, todel, 2)
                    cllabels = np.delete(theta_names, todel)
                    cllabels_raw = np.delete(theta_names_raw, todel)
                    clmostprop = np.delete(mostprop, todel)
                    
                    cldim = len(cllabels)
                    if plot:
                        fig, axes = plt.subplots(cldim, figsize=(8, cldim/1.5),
                                                sharex=True)
                        for i in range(cldim):
                            ax = axes[i]
                            ax.plot(clsamples[:, :, i].T, "k", alpha=0.3)
                            ax.set_ylabel(cllabels[i])
                            ax.yaxis.set_label_coords(-0.1, 0.5)
                        axes[-1].set_xlabel("step number")
                        
                        if createpdf:
                            pdfname = '%s_pol%i_1.png' % (savetime, idx)
                            plt.savefig(pdfname)
                            plt.close()
                        else:
                            plt.show()
                    
                    if nruns > 300:
                        fl_samples = samples[:, -200:, :].reshape((-1, ndim))
                        fl_clsamples = clsamples[:, -200:, :].reshape((-1, cldim))                
                    else:
                        fl_samples = samples.reshape((-1, ndim))
                        fl_clsamples = clsamples.reshape((-1, cldim))

                    if plot:
                        ranges = np.percentile(fl_clsamples, [3, 97], axis=0).T
                        fig = corner.corner(fl_clsamples, quantiles=[0.16, 0.5, 0.84],
                                            truths=clmostprop, labels=cllabels)
                        if createpdf:
                            pdfname = '%s_pol%i_2.png' % (savetime, idx)
                            plt.savefig(pdfname)
                            plt.close()
                        else:
                            plt.show()
                        
                    # get the actual fit
                    theta_fit = np.percentile(fl_samples, [50], axis=0).T
                    if bestchi:
                        theta_result = mostprop
                    else:
                        theta_result = theta_fit
                    
                    fit_visamp, fit_visphi, fit_closure = self.calc_vis(theta_result, u, v, wave, dlambda)
                    fit_vis2 = fit_visamp**2.
                            
                    res_visamp = fit_visamp-visamp
                    res_vis2 = fit_vis2-vis2
                    res_closure_1 = fit_closure-closure
                    res_closure_2 = 360-(fit_closure-closure)
                    check = np.abs(res_closure_1) < np.abs(res_closure_2) 
                    res_closure = res_closure_1*check + res_closure_2*(1-check)
                    res_visphi_1 = fit_visphi-visphi
                    res_visphi_2 = 360-(fit_visphi-visphi)
                    check = np.abs(res_visphi_1) < np.abs(res_visphi_2) 
                    res_visphi = res_visphi_1*check + res_visphi_2*(1-check)

                    redchi_visamp = np.sum(res_visamp**2./visamp_error**2.*(1-visamp_flag))/(visamp.size-np.sum(visamp_flag)-ndof)
                    redchi_vis2 = np.sum(res_vis2**2./vis2_error**2.*(1-vis2_flag))/(vis2.size-np.sum(vis2_flag)-ndof)
                    redchi_closure = np.sum(res_closure**2./closure_error**2.*(1-closure_flag))/(closure.size-np.sum(closure_flag)-ndof)
                    redchi_visphi = np.sum(res_visphi**2./visphi_error**2.*(1-visphi_flag))/(visphi.size-np.sum(visphi_flag)-ndof)
                    redchi = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi]
                    if idx == 0:
                        redchi0 = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi]
                    elif idx == 1:
                        redchi1 = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi]
                        
                    if not bequiet:
                        print('ndof: %i' % (vis2.size-np.sum(vis2_flag)-ndof))
                        print("redchi for visamp: %.2f" % redchi_visamp)
                        print("redchi for vis2: %.2f" % redchi_vis2)
                        print("redchi for closure: %.2f" % redchi_closure)
                        print("redchi for visphi: %.2f" % redchi_visphi)
                        print("average visamp error: %.2f" % 
                            np.mean(visamp_error*(1-visamp_flag)))
                        print("average vis2 error: %.2f" % 
                            np.mean(vis2_error*(1-vis2_flag)))
                        print("average closure error (deg): %.2f" % 
                            np.mean(closure_error*(1-closure_flag)))
                        print("average visphi error (deg): %.2f" % 
                            np.mean(visphi_error*(1-visphi_flag)))
                    

                    percentiles = np.percentile(fl_clsamples, [16, 50, 84],axis=0).T
                    percentiles[:,0] = percentiles[:,1] - percentiles[:,0] 
                    percentiles[:,2] = percentiles[:,2] - percentiles[:,1] 
                    
                    if not bequiet:
                        print("-----------------------------------")
                        print("Best chi2 result:")
                        for i in range(0, cldim):
                            print("%s = %.3f" % (cllabels_raw[i], clmostprop[i]))
                        print("\n")
                        print("MCMC Result:")
                        for i in range(0, cldim):
                            print("%s = %.3f + %.3f - %.3f" % (cllabels_raw[i],
                                                               percentiles[i,1], 
                                                               percentiles[i,2], 
                                                               percentiles[i,0]))
                        print("-----------------------------------")
                    
                    if createpdf:
                        pdf.cell(40, 8, txt="", ln=0, align="L", border="B")
                        pdf.cell(40, 8, txt="Best chi2 result", ln=0, align="L", border="LB")
                        pdf.cell(60, 8, txt="MCMC result", ln=1, align="L", border="LB")
                        for i in range(0, cldim):
                            pdf.cell(40, 6, txt="%s" % cllabels_raw[i], 
                                    ln=0, align="L", border=0)
                            pdf.cell(40, 6, txt="%.3f" % clmostprop[i], 
                                    ln=0, align="C", border="L")
                            pdf.cell(60, 6, txt="%.3f + %.3f - %.3f" % 
                                    (percentiles[i,1], percentiles[i,2], percentiles[i,0]),
                                    ln=1, align="C", border="L")
                        pdf.ln()
                    
                    if plotres and not second_iteration:
                        self.plotFit(theta_result, fitdata, idx, createpdf=createpdf)
                    if write_results:
                        txtfile.write("# Polarization %i  \n" % (idx+1))
                        for tdx, t in enumerate(mostprop):
                            txtfile.write(str(t))
                            txtfile.write(', ')
                        for tdx, t in enumerate(redchi):
                            txtfile.write(str(t))
                            if tdx != (len(redchi)-1):
                                txtfile.write(', ')
                            else:
                                txtfile.write('\n')

                        percentiles = np.percentile(fl_samples, [16, 50, 84],axis=0).T
                        percentiles[:,0] = percentiles[:,1] - percentiles[:,0] 
                        percentiles[:,2] = percentiles[:,2] - percentiles[:,1] 
                        
                        for tdx, t in enumerate(percentiles[:,1]):
                            txtfile.write(str(t))
                            txtfile.write(', ')
                        for tdx, t in enumerate(redchi):
                            txtfile.write(str(t))
                            if tdx != (len(redchi)-1):
                                txtfile.write(', ')
                            else:
                                txtfile.write('\n')

                        for tdx, t in enumerate(percentiles[:,0]):
                            if tdx in todel:
                                txtfile.write(str(t*0.0))
                            else:
                                txtfile.write(str(t))
                            if tdx != (len(percentiles[:,1])-1):
                                txtfile.write(', ')
                            else:
                                txtfile.write(', 0, 0, 0, 0 \n')

                        for tdx, t in enumerate(percentiles[:,2]):
                            if tdx in todel:
                                txtfile.write(str(t*0.0))
                            else:
                                txtfile.write(str(t))
                            if tdx != (len(percentiles[:,1])-1):
                                txtfile.write(', ')
                            else:
                                txtfile.write(', 0, 0, 0, 0 \n')
                                
                    if second_iteration:
                        # Correct the errors to chi2red = 1
                        visamp_error_cor = visamp_error*np.sqrt(redchi_visamp)
                        vis2_error_cor = vis2_error*np.sqrt(redchi_vis2)
                        closure_error_cor = closure_error*np.sqrt(redchi_closure)
                        visphi_error_cor = visphi_error*np.sqrt(redchi_visphi)
                        fitdata_cor = [visamp, visamp_error_cor, visamp_flag,
                                       vis2, vis2_error_cor, vis2_flag,
                                       closure, closure_error_cor, closure_flag,
                                       visphi, visphi_error_cor, visphi_flag]
                        
                        self.fitdata = fitdata_cor
                        # TODO non MCMC version
                        #res = optimize.minimize(self.lnlike_minimize, mostprop, method='BFGS', tol=1e-4, options={'disp': True})
                        
                        # ATTENTION For now only as a second MCMC
                        # ATTENTION slow af
                        width = 1e-1
                        pos = np.ones((nwalkers,ndim))
                        for par in range(ndim):
                            if par in todel:
                                pos[:,par] = mostprop[par]
                            else:
                                pos[:,par] = mostprop[par] + width*np.random.randn(nwalkers)
                                
                        if nthreads == 1:
                            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, 
                                                                args=(fitdata_cor, u, v, wave,
                                                                    dlambda, theta_lower,
                                                                    theta_upper))
                            if bequiet:
                                sampler.run_mcmc(pos, nruns, progress=False)
                            else:
                                sampler.run_mcmc(pos, nruns, progress=True)
                        else:
                            with Pool(processes=nthreads) as pool:
                                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, 
                                                                args=(fitdata_cor, u, v, wave,
                                                                    dlambda, theta_lower,
                                                                    theta_upper),
                                                                pool=pool)
                                if bequiet:
                                    sampler.run_mcmc(pos, nruns, progress=False) 
                                else:
                                    sampler.run_mcmc(pos, nruns, progress=True)  
                        if not bequiet:
                            print("---------------------------------------")
                            print("Mean acceptance fraction: %.2f"  % np.mean(sampler.acceptance_fraction))
                            print("---------------------------------------")
                        samples = sampler.chain
                        mostprop = sampler.flatchain[np.argmax(sampler.flatlnprobability)]

                        clsamples = np.delete(samples, todel, 2)
                        cllabels = np.delete(theta_names, todel)
                        cllabels_raw = np.delete(theta_names_raw, todel)
                        clmostprop = np.delete(mostprop, todel)
                        
                        cldim = len(cllabels)
                        if plot:
                            fig, axes = plt.subplots(cldim, figsize=(8, cldim/1.5),
                                                     sharex=True)
                            for i in range(cldim):
                                ax = axes[i]
                                ax.plot(clsamples[:, :, i].T, "k", alpha=0.3)
                                ax.set_ylabel(cllabels[i])
                                ax.yaxis.set_label_coords(-0.1, 0.5)
                            axes[-1].set_xlabel("step number")
                            
                            if createpdf:
                                pdfname = '%s_pol%i_3.png' % (savetime, idx)
                                plt.savefig(pdfname)
                                plt.close()
                            else:
                                plt.show()
                        
                        if nruns > 300:
                            fl_samples = samples[:, -200:, :].reshape((-1, ndim))
                            fl_clsamples = clsamples[:, -200:, :].reshape((-1, cldim))                
                        else:
                            fl_samples = samples.reshape((-1, ndim))
                            fl_clsamples = clsamples.reshape((-1, cldim))
                        
                        if plot:
                            ranges = np.percentile(fl_clsamples, [3, 97], axis=0).T
                            fig = corner.corner(fl_clsamples, quantiles=[0.16, 0.5, 0.84],
                                                truths=clmostprop, labels=cllabels)
                            if createpdf:
                                pdfname = '%s_pol%i_4.png' % (savetime, idx)
                                plt.savefig(pdfname)
                                plt.close()
                            else:
                                plt.show()
                                
                        # get the actual fit
                        theta_fit = np.percentile(fl_samples, [50], axis=0).T
                        if bestchi:
                            theta_result_cor = mostprop
                        else:
                            theta_result_cor = theta_fit
                        
                        fit_visamp, fit_visphi, fit_closure = self.calc_vis(theta_result_cor, u, v, wave, dlambda)
                        fit_vis2 = fit_visamp**2.
                                
                        res_visamp = fit_visamp-visamp
                        res_vis2 = fit_vis2-vis2
                        res_closure_1 = fit_closure-closure
                        res_closure_2 = 360-(fit_closure-closure)
                        check = np.abs(res_closure_1) < np.abs(res_closure_2) 
                        res_closure = res_closure_1*check + res_closure_2*(1-check)
                        res_visphi_1 = fit_visphi-visphi
                        res_visphi_2 = 360-(fit_visphi-visphi)
                        check = np.abs(res_visphi_1) < np.abs(res_visphi_2) 
                        res_visphi = res_visphi_1*check + res_visphi_2*(1-check)

                        redchi_visamp = np.sum(res_visamp**2./visamp_error_cor**2.*(1-visamp_flag))/(visamp.size-np.sum(visamp_flag)-ndof)
                        redchi_vis2 = np.sum(res_vis2**2./vis2_error_cor**2.*(1-vis2_flag))/(vis2.size-np.sum(vis2_flag)-ndof)
                        redchi_closure = np.sum(res_closure**2./closure_error_cor**2.*(1-closure_flag))/(closure.size-np.sum(closure_flag)-ndof)
                        redchi_visphi = np.sum(res_visphi**2./visphi_error_cor**2.*(1-visphi_flag))/(visphi.size-np.sum(visphi_flag)-ndof)
                        redchi_cor = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi]
                        if idx == 0:
                            redchi0_cor = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi]
                        elif idx == 1:
                            redchi1_cor = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi]
                            
                        if not bequiet:
                            print("redchi for visamp: %.2f" % redchi_visamp)
                            print("redchi for vis2: %.2f" % redchi_vis2)
                            print("redchi for closure: %.2f" % redchi_closure)
                            print("redchi for visphi: %.2f" % redchi_visphi)
                            print("average visamp error: %.2f" % 
                                np.mean(visamp_error_cor*(1-visamp_flag)))
                            print("average vis2 error: %.2f" % 
                                np.mean(vis2_error_cor*(1-vis2_flag)))
                            print("average closure error (deg): %.2f" % 
                                np.mean(closure_error_cor*(1-closure_flag)))
                            print("average visphi error (deg): %.2f" % 
                                np.mean(visphi_error_cor*(1-visphi_flag)))
                        

                        percentiles = np.percentile(fl_clsamples, [16, 50, 84],axis=0).T
                        percentiles[:,0] = percentiles[:,1] - percentiles[:,0] 
                        percentiles[:,2] = percentiles[:,2] - percentiles[:,1] 
                        
                        if not bequiet:
                            print("-----------------------------------")
                            print("Best chi2 result:")
                            for i in range(0, cldim):
                                print("%s = %.3f" % (cllabels_raw[i], clmostprop[i]))
                            print("\n")
                            print("MCMC Result:")
                            for i in range(0, cldim):
                                print("%s = %.3f + %.3f - %.3f" % (cllabels_raw[i], percentiles[i,1], 
                                                                percentiles[i,2], 
                                                                percentiles[i,0]))
                            print("-----------------------------------")
                        if plotres:
                            self.plotFitCor(theta_result, theta_result_cor, fitdata, fitdata_cor, idx, createpdf=createpdf)
                        if write_results:
                            txtfile_cor.write("# Polarization %i  \n" % (idx+1))
                            for tdx, t in enumerate(mostprop):
                                txtfile_cor.write(str(t))
                                txtfile_cor.write(', ')
                            for tdx, t in enumerate(redchi_cor):
                                txtfile_cor.write(str(t))
                                if tdx != (len(redchi_cor)-1):
                                    txtfile_cor.write(', ')
                                else:
                                    txtfile_cor.write('\n')

                            percentiles = np.percentile(fl_samples, [16, 50, 84],axis=0).T
                            percentiles[:,0] = percentiles[:,1] - percentiles[:,0] 
                            percentiles[:,2] = percentiles[:,2] - percentiles[:,1] 
                            
                            for tdx, t in enumerate(percentiles[:,1]):
                                txtfile_cor.write(str(t))
                                txtfile_cor.write(', ')
                            for tdx, t in enumerate(redchi_cor):
                                txtfile_cor.write(str(t))
                                if tdx != (len(redchi_cor)-1):
                                    txtfile_cor.write(', ')
                                else:
                                    txtfile_cor.write('\n')

                            for tdx, t in enumerate(percentiles[:,0]):
                                if tdx in todel:
                                    txtfile_cor.write(str(t*0.0))
                                else:
                                    txtfile_cor.write(str(t))
                                if tdx != (len(percentiles[:,1])-1):
                                    txtfile_cor.write(', ')
                                else:
                                    txtfile_cor.write(', 0, 0, 0, 0 \n')

                            for tdx, t in enumerate(percentiles[:,2]):
                                if tdx in todel:
                                    txtfile_cor.write(str(t*0.0))
                                else:
                                    txtfile_cor.write(str(t))
                                if tdx != (len(percentiles[:,1])-1):
                                    txtfile_cor.write(', ')
                                else:
                                    txtfile_cor.write(', 0, 0, 0, 0 \n')

                
                        

                if createpdf:
                    pdfimages0 = sorted(glob.glob(savetime + '_pol0*.png'))
                    pdfimages1 = sorted(glob.glob(savetime + '_pol1*.png'))
                    pdfcout = 0
                    if plot:
                        pdf.add_page()
                        pdf.cell(0, 10, txt="Polarization  1", ln=1, align="C", border='B')
                        pdf.ln()
                        cover = Image.open(pdfimages0[0])
                        width, height = cover.size
                        ratio = width/height

                        if ratio > (160/115):
                            wi = 160
                            he = 0
                        else:
                            he = 115
                            wi = 0
                        pdf.image(pdfimages0[0], h=he, w=wi)
                        pdf.image(pdfimages0[1], h=115)
                        
                        pdf.add_page()
                        pdf.cell(0, 10, txt="Polarization  2", ln=1, align="C", border='B')
                        pdf.ln()
                        pdf.image(pdfimages1[0], h=he, w=wi)
                        pdf.image(pdfimages1[1], h=115)
                        pdfcout = 2

                        if second_iteration:
                            pdf.add_page()
                            pdf.cell(0, 10, txt="Polarization 1, Iteration 2", ln=1, align="C", border='B')
                            pdf.ln()
                            cover = Image.open(pdfimages0[0])
                            width, height = cover.size
                            ratio = width/height

                            if ratio > (160/115):
                                wi = 160
                                he = 0
                            else:
                                he = 115
                                wi = 0
                            pdf.image(pdfimages0[2], h=he, w=wi)
                            pdf.image(pdfimages0[3], h=115)
                            
                            pdf.add_page()
                            pdf.cell(0, 10, txt="Polarization 2, Iteration 2", ln=1, align="C", border='B')
                            pdf.ln()
                            pdf.image(pdfimages1[2], h=he, w=wi)
                            pdf.image(pdfimages1[3], h=115)
                            pdfcout = 4
                    if plotres:
                        titles = ['Vis Amp', 'Vis 2', 'Closure Phase', 'Visibility Phase']
                        for pa in range(4):
                            pdf.add_page()
                            text = '%s, redchi: %.2f (P1), %.2f (P2)' % (titles[pa], redchi0[pa], redchi1[pa])
                            pdf.cell(0, 10, txt=text, ln=1, align="C", border='B')
                            pdf.ln()
                            pdf.image(pdfimages0[pdfcout+pa], w=150)
                            pdf.image(pdfimages1[pdfcout+pa], w=150)
                    
                    if not bequiet:
                        print('Save pdf as %s' % pdffilename)
                    pdf.output(pdffilename)
                    files = glob.glob(savetime + '_pol?_?.png')
                    for file in files:
                        os.remove(file)
            if write_results:
                txtfile.close()
                if second_iteration:
                    txtfile_cor.close()
        return 0


    def plotFit(self, theta, fitdata, idx=0, createpdf=False):
        """
        Calculates the theoretical interferometric data for the given parameters in theta
        and plots them together with the data in fitdata.
        Mainly used in fitBinary as result plots.
        """
        rad2as = 180 / np.pi * 3600
        
        (visamp, visamp_error, visamp_flag, vis2, 
         vis2_error, vis2_flag, closure, closure_error, 
         closure_flag, visphi, visphi_error, visphi_flag) = fitdata
        wave = self.wlSC_P1
        dlambda = self.dlambda
        wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
        dlambda_model = np.zeros((6,len(wave_model)))
        for i in range(0,6):
            dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
            
        # Fit
        u = self.u
        v = self.v
        magu = np.sqrt(u**2.+v**2.)
        (model_visamp_full, model_visphi_full, 
         model_closure_full)  = self.calc_vis(theta, u, v, wave_model, dlambda_model)
        model_vis2_full = model_visamp_full**2.
        
        u_as = np.zeros((len(u),len(wave)))
        v_as = np.zeros((len(v),len(wave)))
        for i in range(0,len(u)):
            u_as[i,:] = u[i]/(wave*1.e-6) / rad2as
            v_as[i,:] = v[i]/(wave*1.e-6) / rad2as
        magu_as = np.sqrt(u_as**2.+v_as**2.)
        
        u_as_model = np.zeros((len(u),len(wave_model)))
        v_as_model = np.zeros((len(v),len(wave_model)))
        for i in range(0,len(u)):
            u_as_model[i,:] = u[i]/(wave_model*1.e-6) / rad2as
            v_as_model[i,:] = v[i]/(wave_model*1.e-6) / rad2as
        magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)
        
        # Visamp 
        for i in range(0,6):
            plt.errorbar(magu_as[i,:], visamp[i,:]*(1-visamp_flag)[i],
                         visamp_error[i,:]*(1-visamp_flag)[i],
                         color=self.colors_baseline[i],ls='', lw=1, alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], visamp[i,:]*(1-visamp_flag)[i],
                        color=self.colors_baseline[i], alpha=0.5)
            plt.plot(magu_as_model[i,:], model_visamp_full[i,:],
                     color='k', zorder=100)
        plt.ylabel('visibility modulus')
        plt.ylim(-0.1,1.1)
        plt.xlabel('spatial frequency (1/arcsec)')
        if createpdf:
            savetime = self.savetime
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_5.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        # Vis2
        for i in range(0,6):
            plt.errorbar(magu_as[i,:], vis2[i,:]*(1-vis2_flag)[i], 
                         vis2_error[i,:]*(1-vis2_flag)[i], 
                         color=self.colors_baseline[i],ls='', lw=1, alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], vis2[i,:]*(1-vis2_flag)[i],
                        color=self.colors_baseline[i],alpha=0.5)
            plt.plot(magu_as_model[i,:], model_vis2_full[i,:],
                     color='k', zorder=100)
        plt.xlabel('spatial frequency (1/arcsec)')
        plt.ylabel('visibility squared')
        plt.ylim(-0.1,1.1)
        if createpdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_6.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        # T3
        max_u = np.zeros((4))
        max_u[0] = np.max(np.array([magu[0],magu[3],magu[1]]))
        max_u[1] = np.max(np.array([magu[0],magu[4],magu[2]]))
        max_u[2] = np.max(np.array([magu[1],magu[5],magu[2]]))
        max_u[3] = np.max(np.array([magu[3],magu[5],magu[4]]))
        for i in range(0,4):
            max_u_as = max_u[i]/(wave*1.e-6) / rad2as
            max_u_as_model = max_u[i]/(wave_model*1.e-6) / rad2as
            plt.errorbar(max_u_as, closure[i,:]*(1-closure_flag)[i],
                         closure_error[i,:]*(1-closure_flag)[i],
                         color=self.colors_closure[i],ls='', lw=1, alpha=0.5, capsize=0)
            plt.scatter(max_u_as, closure[i,:]*(1-closure_flag)[i],
                        color=self.colors_closure[i], alpha=0.5)
            plt.plot(max_u_as_model, model_closure_full[i,:], 
                     color='k', zorder=100)
        plt.xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
        plt.ylabel('closure phase (deg)')
        if createpdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_7.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        # VisPhi
        for i in range(0,6):
            plt.errorbar(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i], 
                        visphi_error[i,:]*(1-visphi_flag)[i],
                        color=self.colors_baseline[i], ls='', lw=1, alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i],
                        color=self.colors_baseline[i], alpha=0.5)
            plt.plot(magu_as_model[i,:], model_visphi_full[i,:],
                    color='k', zorder=100)
        plt.ylabel('visibility phase')
        plt.xlabel('spatial frequency (1/arcsec)')
        if createpdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_8.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        
    def plotFitCor(self, theta, theta_cor, fitdata, fitdata_cor, idx=0, createpdf=False):
        """
        Calculates the theoretical interferometric data for the given parameters in theta
        and in theta cor, to compare two fits.
        Plots them together with the data in fitdata.
        Mainly used in fitBinary as result plots.
        """
        rad2as = 180 / np.pi * 3600
        
        (visamp, visamp_error, visamp_flag, vis2, 
         vis2_error, vis2_flag, closure, closure_error, 
         closure_flag, visphi, visphi_error, visphi_flag) = fitdata
        (visamp, visamp_error_cor, visamp_flag, vis2, 
         vis2_error_cor, vis2_flag, closure, closure_error_cor, 
         closure_flag, visphi, visphi_error_cor, visphi_flag) = fitdata_cor
        
        
        wave = self.wlSC_P1
        dlambda = self.dlambda
        wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
        dlambda_model = np.zeros((6,len(wave_model)))
        for i in range(0,6):
            dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
            
        # Fit
        u = self.u
        v = self.v
        magu = np.sqrt(u**2.+v**2.)
        (model_visamp_full, model_visphi_full, 
         model_closure_full)  = self.calc_vis(theta, u, v, wave_model, dlambda_model)
        model_vis2_full = model_visamp_full**2.
        (model_visamp_cor, model_visphi_cor, 
         model_closure_cor)  = self.calc_vis(theta_cor, u, v, wave_model, dlambda_model)
        model_vis2_cor = model_visamp_cor**2.
        
        u_as = np.zeros((len(u),len(wave)))
        v_as = np.zeros((len(v),len(wave)))
        for i in range(0,len(u)):
            u_as[i,:] = u[i]/(wave*1.e-6) / rad2as
            v_as[i,:] = v[i]/(wave*1.e-6) / rad2as
        magu_as = np.sqrt(u_as**2.+v_as**2.)
        
        u_as_model = np.zeros((len(u),len(wave_model)))
        v_as_model = np.zeros((len(v),len(wave_model)))
        for i in range(0,len(u)):
            u_as_model[i,:] = u[i]/(wave_model*1.e-6) / rad2as
            v_as_model[i,:] = v[i]/(wave_model*1.e-6) / rad2as
        magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)
        
        # Visamp 
        for i in range(0,6):
            plt.errorbar(magu_as[i,:], visamp[i,:]*(1-visamp_flag)[i],
                         visamp_error[i,:]*(1-visamp_flag)[i],
                         color=self.colors_baseline[i],ls='', lw=2, alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], visamp[i,:]*(1-visamp_flag)[i],
                        color=self.colors_baseline[i], alpha=0.5)
            plt.plot(magu_as_model[i,:], model_visamp_full[i,:],
                     color=self.colors_baseline[i], alpha=1.0)
            plt.errorbar(magu_as[i,:], visamp[i,:]*(1-visamp_flag)[i],
                         visamp_error_cor[i,:]*(1-visamp_flag)[i],
                         color=self.colors_baseline[i],ls='', lw=0.5, marker='', 
                         alpha=1, capsize=0)
            plt.plot(magu_as_model[i,:], model_visamp_cor[i,:],
                     color=self.colors_baseline[i], alpha=1.0, ls='--')
        plt.ylabel('visibility modulus')
        plt.ylim(-0.1,1.1)
        plt.xlabel('spatial frequency (1/arcsec)')
        if createpdf:
            savetime = self.savetime
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_5.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        # Vis2
        for i in range(0,6):
            plt.errorbar(magu_as[i,:], vis2[i,:]*(1-vis2_flag)[i], 
                         vis2_error[i,:]*(1-vis2_flag)[i], 
                         color=self.colors_baseline[i],ls='', lw=2, alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], vis2[i,:]*(1-vis2_flag)[i],
                        color=self.colors_baseline[i],alpha=0.5)
            plt.plot(magu_as_model[i,:], model_vis2_full[i,:],
                     color=self.colors_baseline[i], alpha=1.0)
            plt.errorbar(magu_as[i,:], vis2[i,:]*(1-vis2_flag)[i],
                         vis2_error_cor[i,:]*(1-vis2_flag)[i],
                         color=self.colors_baseline[i], ls='', marker='', lw=0.5, 
                         alpha=1, capsize=0)
            plt.plot(magu_as_model[i,:], model_vis2_cor[i,:],
                     color=self.colors_baseline[i], alpha=1.0, ls='--')
        plt.xlabel('spatial frequency (1/arcsec)')
        plt.ylabel('visibility squared')
        plt.ylim(-0.1,1.1)
        if createpdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_6.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        # T3
        max_u = np.zeros((4))
        max_u[0] = np.max(np.array([magu[0],magu[3],magu[1]]))
        max_u[1] = np.max(np.array([magu[0],magu[4],magu[2]]))
        max_u[2] = np.max(np.array([magu[1],magu[5],magu[2]]))
        max_u[3] = np.max(np.array([magu[3],magu[5],magu[4]]))
        for i in range(0,4):
            max_u_as = max_u[i]/(wave*1.e-6) / rad2as
            max_u_as_model = max_u[i]/(wave_model*1.e-6) / rad2as
            plt.errorbar(max_u_as, closure[i,:]*(1-closure_flag)[i],
                         closure_error[i,:]*(1-closure_flag)[i],
                         color=self.colors_closure[i], ls='', lw=2, alpha=0.5, capsize=0)
            plt.scatter(max_u_as, closure[i,:]*(1-closure_flag)[i],
                        color=self.colors_closure[i], alpha=0.5)
            plt.plot(max_u_as_model, model_closure_full[i,:], 
                     color=self.colors_closure[i])
            plt.errorbar(max_u_as, closure[i,:]*(1-closure_flag)[i],
                         closure_error_cor[i,:]*(1-closure_flag)[i],
                         color=self.colors_closure[i],ls='', lw=0.5, marker='',
                         alpha=1, capsize=0)
            plt.plot(max_u_as_model, model_closure_cor[i,:], 
                     color=self.colors_closure[i], ls='--')
        plt.xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
        plt.ylabel('closure phase (deg)')
        if createpdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_7.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        # VisPhi
        for i in range(0,6):
            plt.errorbar(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i], 
                        visphi_error[i,:]*(1-visphi_flag)[i],
                        color=self.colors_baseline[i], ls='', lw=2, alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i],
                        color=self.colors_baseline[i], alpha=0.5)
            plt.plot(magu_as_model[i,:], model_visphi_full[i,:],
                    color=self.colors_baseline[i])
            plt.errorbar(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i], 
                        visphi_error_cor[i,:]*(1-visphi_flag)[i],
                        color=self.colors_baseline[i], ls='', lw=0.5, marker='',
                        alpha=1, capsize=0)
            plt.plot(magu_as_model[i,:], model_visphi_cor[i,:],
                    color=self.colors_baseline[i], ls='--')
        plt.ylabel('visibility phase')
        plt.xlabel('spatial frequency (1/arcsec)')
        if createpdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_8.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        
        
    ############################################    
    ############################################
    ################ Unary model ###############
    ############################################
    ############################################    
    
    def simulateUnaryphases(self, dRa, dDec, plot=False):
        """
        Test function to se how a given phasecenter shift would look like
        """
        self.fixedBG = True
        self.fixedBH = True
        self.noBG = True
        self.use_opds = False
        self.specialfit = False
        
        rad2as = 180 / np.pi * 3600
        
        nwave = self.channel
        if nwave != 14:
            raise ValueError('Only usable for 14 channels')
        self.getIntdata(plot=False, flag=False)
        u = self.u
        v = self.v
        wave = self.wlSC_P1
        R = np.zeros((6,nwave))
        R[0,:] = [28.9,16.5,15.6,16.8,17.6,16.4,15.7,17.5,18.8,20.1,20.2,20.5,22.0,28.3]
        R[1,:] = [27.2,15.8,14.9,16.0,16.7,15.7,15.1,16.4,18.2,19.6,20.0,20.3,21.7,25.3]
        R[2,:] = [28.3,16.2,15.3,16.7,17.3,16.3,15.7,17.4,18.8,20.2,20.7,21.1,22.4,27.5]
        R[3,:] = [29.1,17.0,15.9,16.6,17.1,16.6,15.8,16.9,18.8,20.5,21.0,21.3,22.0,24.4]
        R[4,:] = [28.8,16.8,16.1,16.7,17.4,16.7,16.1,17.2,19.0,20.5,21.2,21.6,22.2,25.2]
        R[5,:] = [28.0,16.0,15.0,16.2,16.6,15.7,15.3,16.4,17.8,19.3,19.8,20.0,20.8,24.4]
        dlambda = np.zeros((6,nwave))
        for i in range(0,6):
            if (nwave==11) or (nwave==14):
                dlambda[i,:] = wave/R[i,:]/2
            elif nwave==210:
                dlambda[i,:] = wave/500/2
            else:
                dlambda[i,:] = 0.03817
                
        theta = [dRa, dDec, 0, 0, 0]
        visphi = self.calc_vis_unary(theta, u, v, wave, dlambda)
        
        if plot:
            wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
            dlambda_model = np.zeros((6,len(wave_model)))
            for i in range(0,6):
                dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
            model_visphi_full  = self.calc_vis_unary(theta, u, v, wave_model, dlambda_model)
            magu = np.sqrt(u**2.+v**2.)
            u_as = np.zeros((len(u),len(wave)))
            v_as = np.zeros((len(v),len(wave)))
            for i in range(0,len(u)):
                u_as[i,:] = u[i]/(wave*1.e-6) / rad2as
                v_as[i,:] = v[i]/(wave*1.e-6) / rad2as
            magu_as = np.sqrt(u_as**2.+v_as**2.)

            u_as_model = np.zeros((len(u),len(wave_model)))
            v_as_model = np.zeros((len(v),len(wave_model)))
            for i in range(0,len(u)):
                u_as_model[i,:] = u[i]/(wave_model*1.e-6) / rad2as
                v_as_model[i,:] = v[i]/(wave_model*1.e-6) / rad2as
            magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)
            for i in range(0,6):
                plt.errorbar(magu_as[i,:], visphi[i,:], color=self.colors_baseline[i], ls='', marker='o')
                plt.plot(magu_as_model[i,:], model_visphi_full[i,:], color=self.colors_baseline[i],alpha=1.0)
            plt.ylabel('VisPhi')
            plt.xlabel('spatial frequency (1/arcsec)')
            plt.axhline(0, ls='--', lw=0.5)
            maxval = np.max(np.abs(model_visphi_full))
            if maxval < 45:
                maxplot=50
            elif maxval < 95:
                maxplot=100
            else:
                maxplot=180
            plt.ylim(-maxplot, maxplot)
            plt.suptitle('dRa=%.1f, dDec=%.1f' % (theta[0], theta[1]), fontsize=12)
            plt.show()
        return visphi
        
    
    def calc_vis_unary(self, theta, u, v, wave, dlambda):
        mas2rad = 1e-3 / 3600 / 180 * np.pi
        rad2mas = 180 / np.pi * 3600 * 1e3             
        fixedBG = self.fixedBG
        fixedBH = self.fixedBH
        noBG = self.noBG
        use_opds = self.use_opds
        specialfit = self.specialfit
        
        phaseCenterRA = theta[0]
        phaseCenterDEC = theta[1]
        if fixedBH:
            alpha_SgrA = -0.5
        else:
            alpha_SgrA = theta[2]
        if noBG:
            fluxRatioBG = 0
        else:
            fluxRatioBG = theta[3]
        if fixedBG:
            alpha_bg = 3.
        else:
            alpha_bg = theta[4]
        if use_opds:
            opd1 = theta[5]
            opd2 = theta[6]
            opd3 = theta[7]
            opd4 = theta[8]
            opd_bl = np.array([[opd4, opd3],
                               [opd4, opd2],
                               [opd4, opd1],
                               [opd3, opd2],
                               [opd3, opd1],
                               [opd2, opd1]])
        if specialfit:
            special_par = theta[9] 
            sp_bl = np.ones(6)*special_par
            sp_bl *= self.specialfit_bl

        vis = np.zeros((6,len(wave))) + 0j
        if len(u) != 6 or len(v) != 6:
            raise ValueError('u or v have wrong length, something went wrong')                
                
                
        vis = np.zeros((6,len(wave))) + 0j
        for i in range(0,6):
            s_SgrA = ((phaseCenterRA)*u[i] + (phaseCenterDEC)*v[i]) * mas2rad * 1e6
            
            if use_opds:
                s_SgrA = s_SgrA + opd_bl[i,0] - opd_bl[i,1]
            if specialfit:
                s_SgrA = s_SgrA + sp_bl[i]

            # u,v in 1/mas
            u_mas = u[i] / (wave*1e-6) / rad2mas
            v_mas = v[i] / (wave*1e-6) / rad2mas
    
            # interferometric intensities of all components
            intSgrA = self.vis_intensity(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
            intSgrA_center = self.vis_intensity(0, alpha_SgrA, wave, dlambda[i,:])
            intBG = self.vis_intensity(0, alpha_bg, wave, dlambda[i,:])
    
            vis[i,:] = (intSgrA/
                        (intSgrA_center + fluxRatioBG * intBG))
            
        visphi = np.angle(vis, deg=True)
        visphi = visphi + 360.*(visphi<-180.) - 360.*(visphi>180.)
        return visphi   


    def lnlike_unary(self, theta, fitdata, u, v, wave, dlambda):
        model_visphi = self.calc_vis_unary(theta,u,v,wave,dlambda)
        visphi, visphi_error, visphi_flag = fitdata
        res_phi = (-np.minimum((model_visphi-visphi)**2.,
                                     (360-(model_visphi-visphi))**2.)/
                          visphi_error**2.*(1-visphi_flag))
        res_phi = np.sum(res_phi[~np.isnan(res_phi)])
        return 0.5*res_phi
    
    
    def lnprob_unary(self, theta, fitdata, u, v, wave, dlambda, lower, upper):
        if np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        return self.lnlike_unary(theta, fitdata, u, v, wave, dlambda)
    
    
    def plot_unary(self, theta, giveuv=False, uu=None, vv=None, plot=False):
        """
        Test function to see how a given theta would look like in phases.
        Input:  theta = [dRa, dDec]
                giveuv has to be set to True if you wnat to change the uv data
                otherwie it will take them from the class
        """
        theta_names_raw = np.array(["PC RA", "PC DEC"])
        rad2as = 180 / np.pi * 3600
        try:
            if len(theta) != 2:
                print('Theta has to include the following 2 parameter:')
                print(theta_names_raw)
                raise ValueError('Wrong number of input parameter given (should be 2)')
        except(TypeError):
            print('Thetha has to include the following 2 parameter:')
            print(theta_names_raw)
            raise ValueError('Wrong number of input parameter given (should be 2)')
        self.fixedBG = True
        self.fixedBH = True
        self.noBG = True
        self.use_opds = False
        self.specialfit = False
        nwave = self.channel
        if giveuv:
            if uu is None or vv is None:
                raise ValueError("if giveuv=True values for u and v have to be given to the function (as uu, vv)")
            else:
                u = uu
                v = vv
        else:
            u = self.u
            v = self.v
        wave = self.wlSC_P1
        
        
        R = np.zeros((6,nwave))
        R[0,:] = [28.9,16.5,15.6,16.8,17.6,16.4,15.7,17.5,18.8,20.1,20.2,20.5,22.0,28.3]
        R[1,:] = [27.2,15.8,14.9,16.0,16.7,15.7,15.1,16.4,18.2,19.6,20.0,20.3,21.7,25.3]
        R[2,:] = [28.3,16.2,15.3,16.7,17.3,16.3,15.7,17.4,18.8,20.2,20.7,21.1,22.4,27.5]
        R[3,:] = [29.1,17.0,15.9,16.6,17.1,16.6,15.8,16.9,18.8,20.5,21.0,21.3,22.0,24.4]
        R[4,:] = [28.8,16.8,16.1,16.7,17.4,16.7,16.1,17.2,19.0,20.5,21.2,21.6,22.2,25.2]
        R[5,:] = [28.0,16.0,15.0,16.2,16.6,15.7,15.3,16.4,17.8,19.3,19.8,20.0,20.8,24.4]
        dlambda = np.zeros((6,nwave))
        for i in range(0,6):
            if (nwave==11) or (nwave==14):
                dlambda[i,:] = wave/R[i,:]/2
            elif nwave==210:
                dlambda[i,:] = wave/500/2
            else:
                dlambda[i,:] = 0.03817
        visphi = self.calc_vis_unary(theta, u, v, wave, dlambda)
        return visphi
        

    def fitUnary(self, nthreads=4, nwalkers=500, nruns=500, bestchi=True,
                        plot=True, fixedBG=True, fixedBH=True, noBG=True,
                        fitopds=np.array([0,0,0,0]), write_results=True, 
                        flagtill=2, flagfrom=12, plotres=True, createpdf=True, 
                        bequiet=False, noS2=False, mindatapoints=3,
                        dontfit=None, dontfitbl=None, writefitdiff=False,
                        specialpar=np.array([0,0,0,0,0,0])):
        """
        Does a MCMC unary fit on the phases of the data.
        Parameter:
        nthreads:       number of cores [4] 
        nwalkers:       number of walkers [500] 
        nruns:          number of MCMC runs [500] 
        bestchi:        Gives best chi2 (for True) or mcmc res as output [True]
        plot:           plot MCMC results [True]
        fixedBG:        Fit for background power law [False]
        fixedBH:        Fit for black hole power law [False]
        noBG:           Sets BG ratio to 0 [False]
        fitopds:        Fit individual opds id equal 1 [0,0,0,0]
        write_results:  Write fit results in file [True] 
        flagtill:       Flag blue channels [1] 
        flagfrom:       Flag red channels [13]
        plotres:        plot fit result [True]
        createpdf:      Creates a pdf with fit results and all plots [True] 
        bequiet:        Suppresses ALL outputs [False]
        noS2:           If True ignores files where fiber offset is 0 [False]
        mindatapoints:  if less valid datapoints in one baselin, file is rejected [3]
        dontfit:        Number of telescope to flag
        dontfitbl:      Number of baseline to flag
        writefitdiff:   Writes the difference of the mean fit vs data instead of redchi2
        specialfit:     Special version of fit, changing implementaion 
                        (dummy to allow an additional fit parameter)
        """
        rad2as = 180 / np.pi * 3600
        self.fixedBG = fixedBG
        self.fixedBH = fixedBH
        self.noBG = noBG
        if np.any(fitopds):
            self.use_opds = True
            use_opds = True
        else:
            self.use_opds = False
        
        if np.any(specialpar):
            self.specialfit = True
            #self.specialfit_bl = np.array([0,1,1,1,1,0])
            #self.specialfit_bl = np.array([1,0,0,0,0,1])
            #self.specialfit_bl = np.array([1,1,0,0,-1,-1])
            self.specialfit_bl = specialpar
            if not bequiet:
                print('Specialfit parameter applied to BLs:')
                nonzero = np.nonzero(self.specialfit_bl)[0]
                print(*list(nonzero*self.specialfit_bl[nonzero]))
                print('\n')
        else:
            self.specialfit = False
        specialfit = self.specialfit
        # Get data from file
        nwave = self.channel
        self.getIntdata(plot=False, flag=False)
        MJD = fits.open(self.name)[0].header["MJD-OBS"]
        fullu = self.u
        fullv = self.v
        wave = self.wlSC_P1
        self.fiberOffX = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFX"] 
        self.fiberOffY = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFY"] 
        if not bequiet:
            print("fiber center: %.2f, %.2f (mas)" % (self.fiberOffX,
                                                    self.fiberOffY))
        if self.fiberOffX == 0 and self.fiberOffY == 0:
            if noS2:
                if not bequiet:
                    print('No Fiber offset, if you want to fit this file use noS2=False')
                return 0
            
        stname = self.name.find('GRAVI')     
        if dontfit is not None:
            savefilename = 'punaryfit_woUT' + str(dontfit) + '_' + self.name[stname:-5]
        elif dontfitbl is not None:
            if isinstance(dontfitbl, list):
                blname = "".join(str(i) for i in dontfitbl)
                savefilename = 'punaryfit_woBL' + str(blname) + '_' + self.name[stname:-5]
            elif isinstance(dontfitbl, int):
                savefilename = 'punaryfit_woBL' + str(dontfitbl) + '_' + self.name[stname:-5]
                
        else:
            savefilename = 'punaryfit_' + self.name[stname:-5]
        txtfilename = savefilename + '.txt'
        if write_results:
            txtfile = open(txtfilename, 'w')
            txtfile.write('# Results of Unary fit for %s \n' % self.name[stname:])
            txtfile.write('# Lines are: Best chi2, MCMC result, MCMC error -, MCMC error + \n')
            txtfile.write('# Rowes are: PC RA, PC DEC, alpha SgrA*, f BG, alpha BG \n')
            txtfile.write('# Parameter which are not fitted have 0.0 as error \n')
            txtfile.write('# MJD: %f \n' % MJD)
            txtfile.write('# OFFX: %f \n' % self.fiberOffX)
            txtfile.write('# OFFY: %f \n\n' % self.fiberOffY)

        if self.polmode == 'COMBINED':
            R = np.zeros((6,nwave))
            if nwave == 11:
                R[0,:] = [32.9,20.6,20.3,19.3,19.2,16.1,18.3,20.8,21.2,21.7,23.4]
                R[1,:] = [31.8,18.6,17.5,18.5,19.8,16.8,19.8,22.7,22.6,22.8,22.7]
                R[2,:] = [31.8,19.1,19.0,18.7,18.9,16.3,19.1,21.6,22.2,22.5,23.6]
                R[3,:] = [29.9,18.3,18.6,20.6,23.5,19.5,22.7,25.8,25.4,26.8,26.2]
                R[4,:] = [30.8,18.0,17.6,19.3,22.3,19.4,23.3,26.5,26.3,27.7,24.9]
                R[5,:] = [29.7,18.1,18.1,18.1,18.6,16.5,19.6,22.4,22.8,22.8,22.3]
            elif nwave == 14:
                R[0,:] = [28.9,16.5,15.6,16.8,17.6,16.4,15.7,17.5,18.8,20.1,20.2,20.5,22.0,28.3]
                R[1,:] = [27.2,15.8,14.9,16.0,16.7,15.7,15.1,16.4,18.2,19.6,20.0,20.3,21.7,25.3]
                R[2,:] = [28.3,16.2,15.3,16.7,17.3,16.3,15.7,17.4,18.8,20.2,20.7,21.1,22.4,27.5]
                R[3,:] = [29.1,17.0,15.9,16.6,17.1,16.6,15.8,16.9,18.8,20.5,21.0,21.3,22.0,24.4]
                R[4,:] = [28.8,16.8,16.1,16.7,17.4,16.7,16.1,17.2,19.0,20.5,21.2,21.6,22.2,25.2]
                R[5,:] = [28.0,16.0,15.0,16.2,16.6,15.7,15.3,16.4,17.8,19.3,19.8,20.0,20.8,24.4]
        else:
            R = np.zeros((6,nwave))
            if nwave == 11:
                R[0,:] = [25.0,18.3,20.1,20.7,20.0,16.7,17.9,20.1,20.5,21.3,23.9]
                R[1,:] = [24.0,16.5,15.9,17.2,17.6,15.3,17.3,19.7,20.2,20.8,22.4]
                R[2,:] = [26.2,19.5,19.8,19.3,18.9,16.1,17.6,19.9,20.7,21.3,23.4]
                R[3,:] = [24.6,16.4,15.5,17.2,18.2,16.9,19.5,22.2,22.6,23.0,22.5]
                R[4,:] = [26.4,17.5,16.2,17.2,17.9,16.3,18.8,21.4,21.6,21.8,22.1]
                R[5,:] = [27.4,18.8,17.5,17.6,17.4,15.2,16.8,19.2,19.8,20.0,21.1]
            elif nwave == 14:
                R[0,:] = [28.9,16.5,15.6,16.8,17.6,16.4,15.7,17.5,18.8,20.1,20.2,20.5,22.0,28.3]
                R[1,:] = [27.2,15.8,14.9,16.0,16.7,15.7,15.1,16.4,18.2,19.6,20.0,20.3,21.7,25.3]
                R[2,:] = [28.3,16.2,15.3,16.7,17.3,16.3,15.7,17.4,18.8,20.2,20.7,21.1,22.4,27.5]
                R[3,:] = [29.1,17.0,15.9,16.6,17.1,16.6,15.8,16.9,18.8,20.5,21.0,21.3,22.0,24.4]
                R[4,:] = [28.8,16.8,16.1,16.7,17.4,16.7,16.1,17.2,19.0,20.5,21.2,21.6,22.2,25.2]
                R[5,:] = [28.0,16.0,15.0,16.2,16.6,15.7,15.3,16.4,17.8,19.3,19.8,20.0,20.8,24.4]
            
        dlambda = np.zeros((6,nwave))
        for i in range(0,6):
            if (nwave==11) or (nwave==14):
                dlambda[i,:] = wave/R[i,:]/2
            elif nwave==233:
                dlambda[i,:] = wave/500/2
            else:
                dlambda[i,:] = 0.03817
        self.dlambda = dlambda 


        # Initial guesses
        size = 5
        phase_center_RA = 0.1
        phase_center_DEC = 0.1
        phase_center_RA_init = np.array([phase_center_RA,
                                         phase_center_RA - size,
                                         phase_center_RA + size])
        phase_center_DEC_init = np.array([phase_center_DEC,
                                          phase_center_DEC - size,
                                          phase_center_DEC + size])
        alpha_SgrA_init = np.array([-1.,-5.,7.])
        flux_ratio_bg_init = np.array([0.1,0.,20.])
        alpha_bg_init = np.array([3.,-5.,5.])
        opd_max = 1.5 # maximum opd in microns
        opd_1_init = [0.0,-opd_max,opd_max]
        opd_2_init = [0.0,-opd_max,opd_max]
        opd_3_init = [0.0,-opd_max,opd_max]
        opd_4_init = [0.0,-opd_max,opd_max]
        special_par = [-0.15, -2, 2]

        # initial fit parameters 
        theta = np.array([phase_center_RA_init[0], phase_center_DEC_init[0],
                          alpha_SgrA_init[0], flux_ratio_bg_init[0], alpha_bg_init[0],
                          opd_1_init[0],opd_2_init[0],opd_3_init[0],opd_4_init[0],
                          special_par[0]])
        theta_lower = np.array([phase_center_RA_init[1], phase_center_DEC_init[1],
                                alpha_SgrA_init[1], flux_ratio_bg_init[1],
                                alpha_bg_init[1],opd_1_init[1],opd_2_init[1],
                                opd_3_init[1],opd_4_init[1], special_par[1]])
        theta_upper = np.array([phase_center_RA_init[2], phase_center_DEC_init[2],
                                alpha_SgrA_init[2], flux_ratio_bg_init[2],
                                alpha_bg_init[2],opd_1_init[2],opd_2_init[2],
                                opd_3_init[2],opd_4_init[2], special_par[2]])

        theta_names = np.array([r"$RA_{PC}$", r"$DEC_{PC}$", r"$\alpha_{SgrA}$", 
                                r"$f_{bg}$",r"$\alpha_{bg}$", "OPD1", "OPD2", 
                                "OPD3", "OPD4", "special"])
        theta_names_raw = np.array(["PC RA", "PC DEC", "alpha SgrA", "f BG", "alpha BG", 
                                    "OPD1", "OPD2", "OPD3", "OPD4", "special"])

        ndim = len(theta)
        todel = []
        if fixedBH:
            todel.append(2)
        if noBG:
            todel.append(3)
        if fixedBG:
            todel.append(4)
        for tel in range(4):
            if fitopds[tel] != 1:
                todel.append(5+tel)
        if not specialfit:
            todel.append(9)
        ndof = ndim - len(todel)

        # Get data
        if self.polmode == 'SPLIT':
            visphi_P = [self.visphiSC_P1, self.visphiSC_P2]
            visphi_error_P = [self.visphierrSC_P1, self.visphierrSC_P2]
            visphi_flag_P = [self.visampflagSC_P1, self.visampflagSC_P2]
            
            ndit = np.shape(self.visampSC_P1)[0]//6
            if not bequiet:
                print('NDIT = %i' % ndit)
            for dit in range(ndit):
                savetime = str(datetime.now()).replace('-', '')
                savetime = savetime.replace(' ', '-')
                savetime = savetime.replace(':', '')
                self.savetime = savetime
                if write_results and ndit > 1:
                    txtfile.write('# DIT %i \n' % dit)
                if createpdf:
                    savetime = str(datetime.now()).replace('-', '')
                    savetime = savetime.replace(' ', '-')
                    savetime = savetime.replace(':', '')
                    self.savetime = savetime
                    if ndit == 1:
                        pdffilename = savefilename + '.pdf'
                    else:
                        pdffilename = savefilename + '_DIT' + str(dit) + '.pdf'

                    pdf = FPDF(orientation='P', unit='mm', format='A4')
                    pdf.add_page()
                    pdf.set_font("Helvetica", size=12)
                    pdf.set_margins(20,20)
                    if ndit == 1:
                        pdf.cell(0, 10, txt="Fit report for %s" % self.name[stname:], ln=2, align="C", border='B')
                    else:
                        pdf.cell(0, 10, txt="Fit report for %s, dit %i" % (self.name[stname:], dit), ln=2, align="C", border='B')
                    pdf.ln()
                    pdf.cell(40, 6, txt="Fringe Tracker", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=self.header["ESO FT ROBJ NAME"], ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Science Object", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=self.header["ESO INS SOBJ NAME"], ln=1, align="L", border=0)
                    
                    pdf.cell(40, 6, txt="Science Offset X", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(self.header["ESO INS SOBJ OFFX"]), 
                            ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Science Offset Y", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(self.header["ESO INS SOBJ OFFY"]), 
                            ln=1, align="L", border=0)

                    pdf.cell(40, 6, txt="Fixed Bg", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(fixedBG), ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Fixed BH", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(fixedBH), ln=1, align="L", border=0)
                    
                    pdf.cell(40, 6, txt="Flag before/after", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(flagtill) + '/' + str(flagfrom), 
                            ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt="Result: Best Chi2", ln=0, align="L", border=0)
                    pdf.cell(40, 6, txt=str(bestchi), ln=1, align="L", border=0)
                    pdf.ln()

                if not bequiet:
                    print('Run MCMC for DIT %i' % (dit+1))
                ditstart = dit*6
                ditstop = ditstart + 6
                
                bothdofit = np.ones(2)
                for idx in range(2):
                    visphi = visphi_P[idx][ditstart:ditstop]
                    visphi_error = visphi_error_P[idx][ditstart:ditstop]
                    visphi_flag = visphi_flag_P[idx][ditstart:ditstop]
                    u = fullu[ditstart:ditstop]
                    v = fullv[ditstart:ditstop]
                    
                    if ((flagtill > 0) and (flagfrom > 0)):
                        p = flagtill
                        t = flagfrom
                        if idx == 0 and dit == 0:
                            if not bequiet:
                                print('using channels from #%i to #%i' % (p, t))
                        visphi_flag[:,0:p] = True
                        visphi_flag[:,t:] = True
                        
                    # check if the data is good enough to fit
                    dofit = True
                    if (u == 0.).any():
                        if not bequiet:
                            print('some values in u are zero, something wrong in the data')
                        dofit = False
                    if (v == 0.).any():
                        if not bequiet:
                            print('some values in v are zero, something wrong in the data')
                        dofit = False
                    for bl in range(6):
                        if (visphi_flag[bl] == True).all():
                            if not bequiet:
                                print('Baseline %i is completely flagged, something wrong with the data' % bl)
                            dofit = False
                        elif (np.size(visphi_flag[bl])-np.count_nonzero(visphi_flag[bl])) < mindatapoints:
                            if not bequiet:
                                print('Baseline %i is has to few non flagged values' % bl)
                            dofit = False
                    bothdofit[idx] = dofit
                    
                    if dontfit is not None:
                        if not bequiet:
                            print('Will not fit Telescope %i' % dontfit)
                        if dontfit not in [1,2,3,4]:
                            raise ValueError('Dontfit has to be one of the UTs: 1,2,3,4')
                        telescopes = [[4, 3],[4, 2],[4, 1],[3, 2],[3, 1],[2, 1]]
                        for bl in range(6):
                            if dontfit in telescopes[bl]:
                                visphi_flag[bl,:] = True
                    if dontfitbl is not None:
                        if isinstance(dontfitbl, int):
                            if not bequiet:
                                print('Will not fit baseline %i' % dontfitbl)
                            if dontfitbl not in [1,2,3,4,5,6]:
                                raise ValueError('Dontfit has to be one of the UTs: 1,2,3,4,5,6')
                            if dontfit is not None:
                                raise ValueError('Use either dontfit or dontfitbl, not both')
                            visphi_flag[dontfitbl-1,:] = True
                        elif isinstance(dontfitbl, list):
                            for bl in dontfitbl:
                                if not bequiet:
                                    print('Will not fit baseline %i' % bl)
                                if bl not in [1,2,3,4,5,6]:
                                    raise ValueError('Dontfit has to be one of the UTs: 1,2,3,4,5,6')
                                if dontfit is not None:
                                    raise ValueError('Use either dontfit or dontfitbl, not both')
                                visphi_flag[bl-1,:] = True
                                
                            
                        
                            
                            
                    if dofit == True:
                        width = 1e-1
                        pos = np.ones((nwalkers,ndim))
                        for par in range(ndim):
                            if par in todel:
                                pos[:,par] = theta[par]
                            else:
                                pos[:,par] = theta[par] + width*np.random.randn(nwalkers)
                        if not bequiet:
                            print('Run MCMC for Pol %i' % (idx+1))
                        fitdata = [visphi, visphi_error, visphi_flag]
                        if nthreads == 1:
                            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                                            self.lnprob_unary,
                                                            args=(fitdata, u, v, wave,
                                                                dlambda, theta_lower,
                                                                theta_upper))
                            if bequiet:
                                sampler.run_mcmc(pos, nruns, progress=False)
                            else:
                                sampler.run_mcmc(pos, nruns, progress=True)
                        else:
                            with Pool(processes=nthreads) as pool:
                                sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                                self.lnprob_unary, 
                                                                args=(fitdata, u, v, wave,
                                                                    dlambda, theta_lower,
                                                                    theta_upper),
                                                                pool=pool)
                                if bequiet:
                                    sampler.run_mcmc(pos, nruns, progress=False) 
                                else:
                                    sampler.run_mcmc(pos, nruns, progress=True)     
                                
                        if not bequiet:
                            print("---------------------------------------")
                            print("Mean acceptance fraction: %.2f"  % np.mean(sampler.acceptance_fraction))
                            print("---------------------------------------")
                        if createpdf:
                            pdf.cell(0, 10, txt="Polarization  %i" % (idx+1), ln=2, align="C", border='B')
                            pdf.cell(0, 10, txt="Mean acceptance fraction: %.2f"  %
                                    np.mean(sampler.acceptance_fraction), 
                                    ln=2, align="L", border=0)
                        samples = sampler.chain
                        mostprop = sampler.flatchain[np.argmax(sampler.flatlnprobability)]

                        clsamples = np.delete(samples, todel, 2)
                        cllabels = np.delete(theta_names, todel)
                        cllabels_raw = np.delete(theta_names_raw, todel)
                        clmostprop = np.delete(mostprop, todel)
                            
                        cldim = len(cllabels)
                        if plot:
                            fig, axes = plt.subplots(cldim, figsize=(8, cldim/1.5),
                                                    sharex=True)
                            for i in range(cldim):
                                ax = axes[i]
                                ax.plot(clsamples[:, :, i].T, "k", alpha=0.3)
                                ax.set_ylabel(cllabels[i])
                                ax.yaxis.set_label_coords(-0.1, 0.5)
                            axes[-1].set_xlabel("step number")
                                
                            if createpdf:
                                pdfname = '%s_pol%i_1.png' % (savetime, idx)
                                plt.savefig(pdfname)
                                plt.close()
                            else:
                                plt.show()
                            
                        if nruns > 300:
                            fl_samples = samples[:, -200:, :].reshape((-1, ndim))
                            fl_clsamples = clsamples[:, -200:, :].reshape((-1, cldim))                
                        elif nruns > 200:
                            fl_samples = samples[:, -100:, :].reshape((-1, ndim))
                            fl_clsamples = clsamples[:, -100:, :].reshape((-1, cldim))                
                        else:
                            fl_samples = samples.reshape((-1, ndim))
                            fl_clsamples = clsamples.reshape((-1, cldim))

                        if plot:
                            ranges = np.percentile(fl_clsamples, [3, 97], axis=0).T
                            fig = corner.corner(fl_clsamples, quantiles=[0.16, 0.5, 0.84],
                                                truths=clmostprop, labels=cllabels)
                            if createpdf:
                                pdfname = '%s_pol%i_2.png' % (savetime, idx)
                                plt.savefig(pdfname)
                                plt.close()
                            else:
                                plt.show()
                                
                        # get the actual fit
                        theta_fit = np.percentile(fl_samples, [50], axis=0).T
                        if bestchi:
                            theta_result = mostprop
                        else:
                            theta_result = theta_fit
                            
                        fit_visphi = self.calc_vis_unary(theta_result, u, v, wave, dlambda)
                                    
                        res_visphi_1 = fit_visphi-visphi
                        res_visphi_2 = 360-(fit_visphi-visphi)
                        check = np.abs(res_visphi_1) < np.abs(res_visphi_2) 
                        res_visphi = res_visphi_1*check + res_visphi_2*(1-check)

                        redchi_visphi = np.sum((res_visphi**2./visphi_error**2.)*(1-visphi_flag))/(visphi.size-np.sum(visphi_flag)-ndof)
                        #redchi_visphi = np.sum(res_visphi**2.*(1-visphi_flag))/(visphi.size-np.sum(visphi_flag)-ndof)
                            
                        fitdiff = []
                        for bl in range(6):
                            data = visphi[bl]
                            data[np.where(visphi_flag[bl] == True)] = np.nan
                            fit = fit_visphi[bl]
                            fit[np.where(visphi_flag[bl] == True)] = np.nan
                            err = visphi_error[bl]
                            err[np.where(visphi_flag[bl] == True)] = np.nan              
                            fitdiff.append(np.abs(np.nanmedian(data)-np.nanmedian(fit)))
                            #fitdiff.append(np.abs(np.nanmedian(data)-np.nanmedian(fit))/np.nanmean(err))
                        fitdiff = np.sum(fitdiff)
                            
                        if idx == 0:
                            redchi0 = redchi_visphi
                        elif idx == 1:
                            redchi1 = redchi_visphi
                                
                        if not bequiet:
                            print('ndof: %i' % (visphi.size-np.sum(visphi_flag)-ndof))
                            print("redchi for visphi: %.2f" % redchi_visphi)
                            print("mean fit difference: %.2f" % fitdiff)
                            print("average visphi error (deg): %.2f" % 
                                    np.mean(visphi_error*(1-visphi_flag)))
                            
                        percentiles = np.percentile(fl_clsamples, [16, 50, 84],axis=0).T
                        percentiles[:,0] = percentiles[:,1] - percentiles[:,0] 
                        percentiles[:,2] = percentiles[:,2] - percentiles[:,1] 
                            
                        if not bequiet:
                            print("-----------------------------------")
                            print("Best chi2 result:")
                            for i in range(0, cldim):
                                print("%s = %.3f" % (cllabels_raw[i], clmostprop[i]))
                            print("\n")
                            print("MCMC Result:")
                            for i in range(0, cldim):
                                print("%s = %.3f + %.3f - %.3f" % (cllabels_raw[i],
                                                                    percentiles[i,1], 
                                                                    percentiles[i,2], 
                                                                    percentiles[i,0]))
                            print("-----------------------------------")
                        
                        if createpdf:
                            pdf.cell(40, 8, txt="", ln=0, align="L", border="B")
                            pdf.cell(40, 8, txt="Best chi2 result", ln=0, align="L", border="LB")
                            pdf.cell(60, 8, txt="MCMC result", ln=1, align="L", border="LB")
                            for i in range(0, cldim):
                                pdf.cell(40, 6, txt="%s" % cllabels_raw[i], 
                                        ln=0, align="L", border=0)
                                pdf.cell(40, 6, txt="%.3f" % clmostprop[i], 
                                        ln=0, align="C", border="L")
                                pdf.cell(60, 6, txt="%.3f + %.3f - %.3f" % 
                                        (percentiles[i,1], percentiles[i,2], percentiles[i,0]),
                                        ln=1, align="C", border="L")
                            pdf.ln()
                            
                        if plotres:
                            self.plotFitUnary(theta_result, fitdata, u, v, idx, 
                                            createpdf=createpdf)
                    else:
                        fitdiff = 0
                        redchi_visphi = 0
                    if write_results:
                        if writefitdiff:
                            fitqual = fitdiff
                        else:
                            fitqual = redchi_visphi
                        txtfile.write("# Polarization %i  \n" % (idx+1))
                        if dofit == True:
                            for tdx, t in enumerate(mostprop):
                                txtfile.write(str(t))
                                txtfile.write(', ')
                            txtfile.write(str(fitqual))
                            txtfile.write('\n')
                                    
                            percentiles = np.percentile(fl_samples, [16, 50, 84],axis=0).T
                            percentiles[:,0] = percentiles[:,1] - percentiles[:,0] 
                            percentiles[:,2] = percentiles[:,2] - percentiles[:,1] 
                            
                            for tdx, t in enumerate(percentiles[:,1]):
                                txtfile.write(str(t))
                                txtfile.write(', ')
                            txtfile.write(str(fitqual))
                            txtfile.write('\n')

                            for tdx, t in enumerate(percentiles[:,0]):
                                if tdx in todel:
                                    txtfile.write(str(t*0.0))
                                else:
                                    txtfile.write(str(t))
                                if tdx != (len(percentiles[:,1])-1):
                                    txtfile.write(', ')
                                else:
                                    txtfile.write(', 0 \n')

                            for tdx, t in enumerate(percentiles[:,2]):
                                if tdx in todel:
                                    txtfile.write(str(t*0.0))
                                else:
                                    txtfile.write(str(t))
                                if tdx != (len(percentiles[:,1])-1):
                                    txtfile.write(', ')
                                else:
                                    txtfile.write(', 0 \n')
                        else:
                            nantxt = 'nan, '
                            txtfile.write(nantxt*(ndim) + 'nan \n')
                            txtfile.write(nantxt*(ndim) + 'nan \n')
                            txtfile.write(nantxt*(ndim) + 'nan \n')
                            txtfile.write(nantxt*(ndim) + 'nan \n')
                    
                if createpdf:
                    if (bothdofit == True).all():
                        pdfimages0 = sorted(glob.glob(savetime + '_pol0*.png'))
                        pdfimages1 = sorted(glob.glob(savetime + '_pol1*.png'))
                        pdfcout = 0
                        if plot:
                            pdf.add_page()
                            pdf.cell(0, 10, txt="Polarization  1", ln=1, align="C", border='B')
                            pdf.ln()
                            cover = Image.open(pdfimages0[0])
                            width, height = cover.size
                            ratio = width/height

                            if ratio > (160/115):
                                wi = 160
                                he = 0
                            else:
                                he = 115
                                wi = 0
                            pdf.image(pdfimages0[0], h=he, w=wi)
                            pdf.image(pdfimages0[1], h=115)
                            
                            pdf.add_page()
                            pdf.cell(0, 10, txt="Polarization  2", ln=1, align="C", border='B')
                            pdf.ln()
                            pdf.image(pdfimages1[0], h=he, w=wi)
                            pdf.image(pdfimages1[1], h=115)
                            pdfcout = 2

                        if plotres:
                            titles = ['Visibility Phase']
                            for pa in range(1):
                                pdf.add_page()
                                text = '%s, redchi: %.2f (P1), %.2f (P2)' % (titles[pa], 
                                                                            redchi0, 
                                                                            redchi1)
                                pdf.cell(0, 10, txt=text, ln=1, align="C", border='B')
                                pdf.ln()
                                pdf.image(pdfimages0[pdfcout+pa], w=150)
                                pdf.image(pdfimages1[pdfcout+pa], w=150)
                        
                        if not bequiet:
                            print('Save pdf as %s' % pdffilename)
                        pdf.output(pdffilename)
                    else:
                        del pdf 
                    files = glob.glob(savetime + '_pol?_?.png')
                    for file in files:
                        os.remove(file)
            if write_results:
                txtfile.close()
        return 0
    
    
    
    
            
    def plotFitUnary(self, theta,  fitdata, u, v, idx=0, createpdf=False):
        rad2as = 180 / np.pi * 3600
        (visphi, visphi_error, visphi_flag) = fitdata
        visphi[np.isnan(visphi)] = 0
        wave = self.wlSC_P1
        dlambda = self.dlambda
        savetime = self.savetime
        wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
        dlambda_model = np.zeros((6,len(wave_model)))
        for i in range(0,6):
            dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
            
        # Fit
        magu = np.sqrt(u**2.+v**2.)
        model_visphi_full = self.calc_vis_unary(theta, u, v, wave_model, dlambda_model)

        u_as = np.zeros((len(u),len(wave)))
        v_as = np.zeros((len(v),len(wave)))
        for i in range(0,len(u)):
            u_as[i,:] = u[i]/(wave*1.e-6) / rad2as
            v_as[i,:] = v[i]/(wave*1.e-6) / rad2as
        magu_as = np.sqrt(u_as**2.+v_as**2.)
        
        u_as_model = np.zeros((len(u),len(wave_model)))
        v_as_model = np.zeros((len(v),len(wave_model)))
        for i in range(0,len(u)):
            u_as_model[i,:] = u[i]/(wave_model*1.e-6) / rad2as
            v_as_model[i,:] = v[i]/(wave_model*1.e-6) / rad2as
        magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)

        # VisPhi
        for i in range(0,6):
            plt.errorbar(magu_as[i,:], visphi[i,:]*(1-visphi_flag[i]), 
                        visphi_error[i,:]*(1-visphi_flag[i]), label=self.baseline_labels[i],
                        color=self.colors_baseline[i], ls='', lw=1, alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], visphi[i,:]*(1-visphi_flag[i]),
                        color=self.colors_baseline[i], alpha=0.5)
            plt.plot(magu_as_model[i,:], model_visphi_full[i,:],
                    color='k', zorder=100)
        plt.ylabel('visibility phase')
        plt.xlabel('spatial frequency (1/arcsec)')
        plt.legend()
        if createpdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_8.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
            
