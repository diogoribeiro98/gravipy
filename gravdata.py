from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import emcee
import corner
from multiprocessing import Pool
from fpdf import FPDF
from PIL import Image

from generalFunctions import *
set_style('show')

import sys
import os 



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


#def MCMCprob(theta, fitdata, u, v, wave, dlambda, lower, upper):
    #return 1




class GravData():
    def __init__(self, data, verbose=True):
        self.name = data
        self.verbose = verbose
        
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
        
        self.colors_baseline = np.array(["magenta","crimson","cyan","green","blue","orange"])
        self.colors_closure = np.array(["blue","crimson","cyan","green"])
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
                    
                
                if plot:
                    for idx in range(len(self.vis2SC_P1)):
                        plt.errorbar(self.spFrequAS[idx,:], self.vis2SC_P1[idx,:],self.vis2errSC_P1[idx,:], ls='', marker='o',color=self.colors_baseline[idx%6])
                    for idx in range(len(self.vis2SC_P2)):
                        plt.errorbar(self.spFrequAS[idx,:], self.vis2SC_P2[idx,:],self.vis2errSC_P2[idx,:], ls='', marker='p',color=self.colors_baseline[idx%6])
                    plt.ylim(-0.1,1.4)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('visibility squared')
                    plt.show()

                    for idx in range(len(self.vis2SC_P1)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visphiSC_P1[idx,:],self.visphierrSC_P1[idx,:], ls='', marker='o',color=self.colors_baseline[idx%6])
                    for idx in range(len(self.vis2SC_P2)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visphiSC_P2[idx,:],self.visphierrSC_P2[idx,:], ls='', marker='p',color=self.colors_baseline[idx%6])
                    #plt.ylim(-0.1,1.4)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('visibility phase')
                    plt.show()
                    
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
                    plt.xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
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
    












    def calc_vis(self, theta, u, v, wave, dlambda):
        mas2rad = 1e-3 / 3600 / 180 * np.pi
        rad2mas = 180 / np.pi * 3600 * 1e3

        def intensity(s, alpha, lambda0, dlambda):
            """
            Modulated interferometric intensity
            s = B*skypos-opd1-opd2
            alphs = power law
            """
            return (lambda0/2.2)**(-1-alpha)*2*dlambda*np.sinc(s*2*dlambda/lambda0**2.)*np.exp(-2.j*np.pi*s/lambda0)

        use_coupling = self.use_coupling 
        constant_f = self.constant_f
        fixedBG = self.fixedBG
        use_opds = self.use_opds
        use_visscale = self.use_visscale
        fiberOffX = self.fiberOffX
        fiberOffY = self.fiberOffY
        
        dRA = theta[0]
        dDEC = theta[1]
        if constant_f:
            fluxRatio = theta[2]
        else:
            fluxRatio1 = theta[2]
            fluxRatio2 = theta[3]
            fluxRatio3 = theta[4]
            fluxRatio4 = theta[5]
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

        alpha_S2 = 3
        
        # Flux Ratios
        if use_coupling:
            # TODO can I remove this? Commented in Idels file
            raise ValueError('Coupling ratios not available!')
            f = 10.**fluxRatio1*np.array([coupling_1, coupling_2, coupling_3, coupling_4])
        else:
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
            # s = bl*(sky position) + opd1 - opd2  in mum
            s_SgrA = ((fiberOffX-dRA)*u[i]+(fiberOffY-dDEC)*v[i]) * mas2rad * 1e6
            s_S2 = (fiberOffX*u[i]+fiberOffY*v[i]) * mas2rad * 1e6
            if use_opds:
                s_S2 = s_S2 + opd_bl[i,0] - opd_bl[i,1]
    
            # u,v in 1/mas
            u_mas = u[i]/(wave*1e-6) / rad2mas
            v_mas = v[i]/(wave*1e-6) / rad2mas
            
            # interferometric intensities of all components
            intSgrA = intensity(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
            intS2   = intensity(s_S2, alpha_S2, wave, dlambda[i,:])
            intSgrA_center = intensity(0, alpha_SgrA, wave, dlambda[i,:])
            intS2_center = intensity(0, alpha_S2, wave, dlambda[i,:])
            intBG = intensity(0, alpha_bg, wave, dlambda[i,:])
            
            vis[i,:] = ((intSgrA + 
                         np.sqrt(f_bl[i,0] * f_bl[i,1]) * intS2)/
                        (np.sqrt(intSgrA_center + f_bl[i,0] * intS2_center 
                                 + fluxRatioBG * intBG) *
                         np.sqrt(intSgrA_center + f_bl[i,1] * intS2_center 
                                 + fluxRatioBG * intBG)) *
                         np.exp(-2j*np.pi*(u_mas * phaseCenterRA 
                                           + v_mas * phaseCenterDEC)))
        
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

    
    def fitBinary(self, nthreads=4, nwalkers=500, nruns=500, bestchi=True,
                  plot=True, fit_for=np.array([0.5,0.5,1.0,0.0]), constant_f=True,
                  use_coupling=False, use_opds=False, fixedBG=True,
                  use_visscale=False, write_results=True, flagtill=3, flagfrom=13,
                  dRA=0., dDEC=0., plotres=True, pdf=True):
        '''
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
        use_coupling:   user theoretical coupling [False] 
        use_opds:       Fit OPDs [False] 
        fixedBG:        Fir for background power law [False]
        use_visscale:   Fit for a scaling in visamp [False] 
        flagtill:       Flag blue channels [3] 
        flagfrom:       Flag red channels [13]
        dRA:            Guess for dRA (taken from SOFFX if not 0)
        dDEC:           Guess for dDEC (taken from SOFFY if not 0)
        '''
        stname = self.name.find('GRAVI')        
        pdffilename = 'binaryfit_' + self.name[stname:-5] + '.pdf'
        txtfilename = 'binaryfit_' + self.name[stname:-5] + '.txt'
        
        if write_results:
            txtfile = open(txtfilename, 'w')
            txtfile.write('# Results of binary fit for %s \n' % self.name[stname:])
            txtfile.write('# Lines are: Best chi1, MCMC result, MCMC error 1, MCMC error 1 \n')
            txtfile.write('# Rowes are: dRA, dDEC, f1, f2, f3, f4, alpha flare, V scale, f BG, alpha BG, PC RA, PC DEC, OPD1, OPD2, OPD3, OPD4 \n')
        
        if pdf:
            pdf = FPDF(orientation='P', unit='mm', format='A4')
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.set_margins(20,20)
            pdf.cell(0, 10, txt="Fit report for %s" % self.name[stname:], ln=2, align="C", border='B')
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
            
            pdf.ln()
            
        
        self.fit_for = fit_for
        self.use_coupling = use_coupling
        self.constant_f = constant_f
        self.use_opds = use_opds
        self.fixedBG = fixedBG
        self.use_visscale = use_visscale
        
        # Get data from file
        nwave = self.channel
        self.getIntdata(plot=False, flag=False)
        MJD = fits.open(self.name)[0].header["MJD-OBS"]
        u = self.u
        v = self.v
        wave = self.wlSC_P1
        self.fiberOffX = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFX"] 
        self.fiberOffY = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFY"] 
        print("fiber center: %.2f, %.2f (mas): " % (self.fiberOffX,
                                                    self.fiberOffY))
        if self.fiberOffX != 0 and self.fiberOffY != 0:
            dRA = self.fiberOffX
            dDEC = self.fiberOffY
        if dRA ==0 and dDEC == 0:
            print('Fiber on S2, guess for dRA & dDEC should be given with function')
            
        
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

                
        # Initial guesses
        size = 2
        dRA_init = np.array([dRA,dRA-size,dRA+size])
        dDEC_init = np.array([dDEC,dDEC-size,dDEC+size])

        fr_start = np.log10(2.0)
        flux_ratio_1_init = np.array([fr_start,np.log10(0.01),np.log10(100.)])
        flux_ratio_2_init = np.array([fr_start,np.log10(0.01),np.log10(100.)])
        flux_ratio_3_init = np.array([fr_start,np.log10(0.01),np.log10(100.)])
        flux_ratio_4_init = np.array([fr_start,np.log10(0.01),np.log10(100.)])

        alpha_SgrA_init = np.array([-1.,-5.,5.])

        vis_scale_init = np.array([0.8,0.1,1.2])

        flux_ratio_bg_init = np.array([0.1,0.,20.])

        color_bg_init = np.array([3.,-5.,5.])

        size = 5
        phase_center_RA = 0.1
        phase_center_DEC = 0.1

        phase_center_RA_init = np.array([phase_center_RA,phase_center_RA-size,phase_center_RA+size])
        phase_center_DEC_init = np.array([phase_center_DEC,phase_center_DEC-size,phase_center_DEC+size])

        opd_max = 0.5 # maximum opd in microns (lambda/4)
        opd_1_init = [0.1,-opd_max,opd_max]
        opd_2_init = [0.1,-opd_max,opd_max]
        opd_3_init = [0.1,-opd_max,opd_max]
        opd_4_init = [0.1,-opd_max,opd_max]

        # initial fit parameters 
        theta = np.array([dRA_init[0],dDEC_init[0],flux_ratio_1_init[0],flux_ratio_2_init[0],flux_ratio_3_init[0],flux_ratio_4_init[0],alpha_SgrA_init[0],vis_scale_init[0],flux_ratio_bg_init[0],color_bg_init[0],phase_center_RA_init[0],phase_center_DEC_init[0],opd_1_init[0],opd_2_init[0],opd_3_init[0],opd_4_init[0]])

        # lower limit on fit parameters 
        theta_lower = np.array([dRA_init[1],dDEC_init[1],flux_ratio_1_init[1],flux_ratio_2_init[1],flux_ratio_3_init[1],flux_ratio_4_init[1],alpha_SgrA_init[1],vis_scale_init[1],flux_ratio_bg_init[1],color_bg_init[1],phase_center_RA_init[1],phase_center_DEC_init[1],opd_1_init[1],opd_2_init[1],opd_3_init[1],opd_4_init[1]])

        # upper limit on fit parameters 
        theta_upper = np.array([dRA_init[2],dDEC_init[2],flux_ratio_1_init[2],flux_ratio_2_init[2],flux_ratio_3_init[2],flux_ratio_4_init[2],alpha_SgrA_init[2],vis_scale_init[2],flux_ratio_bg_init[2],color_bg_init[2],phase_center_RA_init[2],phase_center_DEC_init[2],opd_1_init[2],opd_2_init[2],opd_3_init[2],opd_4_init[2]])

        theta_names = np.array(["dRA", "dDEC", "f1", "f2", "f3", "f4" ,
                                r"$\alpha_{flare}$", r"|V| sc", r"$f_{bg}$",
                                r"$\alpha_{bg}$", r"$RA_{PC}$", r"$DEC_{PC}$",
                                "OPD1", "OPD2", "OPD3", "OPD4"])
        theta_names_raw = np.array(["dRA", "dDEC", "f1", "f2", "f3", "f4" ,
                                    "alpha flare$", "V scale", "f BG",
                                    "alpha BG$", "PC RA", "PC DEC$",
                                    "OPD1", "OPD2", "OPD3", "OPD4"])

        ndof = (5 + 3*np.invert(constant_f) + np.sum(fit_for != 0)*2 + 4*use_opds +
                1*use_visscale + 1*np.invert(fixedBG))

        ndim = len(theta)
                
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
            
            for idx in range(2):
                visamp = visamp_P[idx]
                visamp_error = visamp_error_P[idx]
                visamp_flag = visamp_flag_P[idx]
                vis2 = vis2_P[idx]
                vis2_error = vis2_error_P[idx]
                vis2_flag = vis2_flag_P[idx]
                closure = closure_P[idx]
                closure_error = closure_error_P[idx]
                closure_flag = closure_flag_P[idx]
                visphi = visphi_P[idx]
                visphi_error = visphi_error_P[idx]
                visphi_flag = visphi_flag_P[idx]

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
                    if idx == 0:
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
                    pos[:,par] = theta[par] + width*np.random.randn(nwalkers)

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
                    sampler.run_mcmc(pos, nruns, progress=True)
                else:
                    with Pool(processes=nthreads) as pool:
                        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, 
                                                        args=(fitdata, u, v, wave,
                                                              dlambda, theta_lower,
                                                              theta_upper),
                                                        pool=pool)
                        sampler.run_mcmc(pos, nruns, progress=True)     
                        
                print("---------------------------------------")
                print("Mean acceptance fraction: %.2f"  % np.mean(sampler.acceptance_fraction))
                print("---------------------------------------")
                if pdf:
                    pdf.cell(0, 10, txt="Polarization  %i" % (idx+1), ln=2, align="C", border='B')
                    pdf.cell(0, 10, txt="Mean acceptance fraction: %.2f"  %
                             np.mean(sampler.acceptance_fraction), 
                             ln=2, align="L", border=0)

                samples = sampler.chain
                mostprop = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
                
                todel = []
                if constant_f:
                    todel.append(3)
                    todel.append(4)
                    todel.append(5)
                if not use_visscale:
                    todel.append(7)
                if fixedBG:
                    todel.append(9)
                if not use_opds:
                    todel.append(12)
                    todel.append(13)
                    todel.append(14)
                    todel.append(15)
                    
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
                    
                    if pdf:
                        pdfname = 'temp_pol%i_1.png' % idx
                        plt.savefig(pdfname)
                        plt.close()
                    else:
                        plt.show()
                
                if nruns > 500:
                    show = nruns-150
                    fl_samples = samples[:, show:, :].reshape((-1, ndim))
                    fl_clsamples = clsamples[:, show:, :].reshape((-1, cldim))                
                elif nruns > 300:
                    show = nruns//3*2
                    fl_samples = samples[:, show:, :].reshape((-1, ndim))
                    fl_clsamples = clsamples[:, show:, :].reshape((-1, cldim))
                else:
                    fl_samples = samples.reshape((-1, ndim))
                    fl_clsamples = clsamples.reshape((-1, cldim))

                
                if plot:
                    ranges = np.percentile(fl_clsamples, [1, 99], axis=0).T
                    fig = corner.corner(fl_clsamples, quantiles=[0.16, 0.5, 0.84],
                                        truths=clmostprop, labels=cllabels)
                    if pdf:
                        pdfname = 'temp_pol%i_2.png' % idx
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
                
                
                magu = np.sqrt(u**2.+v**2.) # projected baseline length in meters
                pa = np.arctan2(v,u)
                for i in range(0,len(pa)):
                    if pa[i]<0.:
                        pa[i] += 2.*np.pi
                fit_visamp, fit_visphi, fit_closure = self.calc_vis(theta_result, u, v, 
                                                                    wave, dlambda)
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
                
                if pdf:
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
                    self.plotFit(theta_result, fitdata, idx, pdf=pdf)
                if write_results:
                    txtfile.write("# Polarization %i  \n" % (idx+1))
                    txt = ""
                    for t in mostprop:
                        txtfile.write(str(t))
                        if t != mostprop[-1]:
                            txtfile.write(', ')
                        else:
                            txtfile.write('\n')

                    percentiles = np.percentile(fl_samples, [16, 50, 84],axis=0).T
                    percentiles[:,0] = percentiles[:,1] - percentiles[:,0] 
                    percentiles[:,2] = percentiles[:,2] - percentiles[:,1] 
                    
                    for t in percentiles[:,1]:
                        txtfile.write(str(t))
                        if t != percentiles[-1,1]:
                            txtfile.write(', ')
                        else:
                            txtfile.write('\n')

                    for t in percentiles[:,0]:
                        txtfile.write(str(t))
                        if t != percentiles[-1,0]:
                            txtfile.write(', ')
                        else:
                            txtfile.write('\n')

                    for t in percentiles[:,2]:
                        txtfile.write(str(t))
                        if t != percentiles[-1,2]:
                            txtfile.write(', ')
                        else:
                            txtfile.write('\n')
                    


        
        if pdf:
            pdfimages0 = sorted(glob.glob('temp_pol0*.png'))
            pdfimages1 = sorted(glob.glob('temp_pol1*.png'))
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
                titles = ['Vis Amp', 'Vis 2', 'Closure Phase', 'Visibility Phase']
                for pa in range(4):
                    pdf.add_page()
                    pdf.cell(0, 10, txt=titles[pa], ln=1, align="C", border='B')
                    pdf.ln()
                    pdf.image(pdfimages0[pdfcout+pa], w=150)
                    pdf.image(pdfimages1[pdfcout+pa], w=150)
            
            print('Save pdf as %s' % pdffilename)
            pdf.output(pdffilename)
            files = glob.glob('temp_pol?_?.png')
            for file in files:
                os.remove(file)
        if write_results:
            txtfile.close()
                
        return 0


    def plotFit(self, theta, fitdata, idx=0, pdf=False):
        colors_baseline = np.array(["magenta","crimson","cyan","green","blue","orange"])
        colors_closure = np.array(["blue","crimson","cyan","green"])
        baseline_labels = np.array(["UT4-3","UT4-2","UT4-1","UT3-2","UT3-1","UT2-1"])
        closure_labels = np.array(["UT4-3-2","UT4-3-1","UT4-2-1","UT3-2-1"])


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
        u_dot_sigma = u_as/1000.*theta[0] + v_as/1000.*theta[1]
        
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
                         color=colors_baseline[i],ls='', alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], visamp[i,:]*(1-visamp_flag)[i],
                        color=colors_baseline[i], alpha=0.5)
            plt.plot(magu_as_model[i,:], model_visamp_full[i,:],
                     color=colors_baseline[i], alpha=1.0)
        plt.ylabel('visibility modulus')
        plt.ylim(-0.1,1.1)
        plt.xlabel('spatial frequency (1/arcsec)')
        if pdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = 'temp_pol%i_3.png' % idx
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        # Vis2
        for i in range(0,6):
            plt.errorbar(magu_as[i,:], vis2[i,:]*(1-vis2_flag)[i], 
                         vis2_error[i,:]*(1-vis2_flag)[i], 
                         color=colors_baseline[i],ls='', alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], vis2[i,:]*(1-vis2_flag)[i],
                        color=colors_baseline[i],alpha=0.5)
            plt.plot(magu_as_model[i,:], model_vis2_full[i,:],
                     color=colors_baseline[i], alpha=1.0)
        plt.xlabel('spatial frequency (1/arcsec)')
        plt.ylabel('visibility squared')
        plt.ylim(-0.1,1.1)
        if pdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = 'temp_pol%i_4.png' % idx
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
                         color=colors_closure[i],ls='', alpha=0.5, capsize=0)
            plt.scatter(max_u_as, closure[i,:]*(1-closure_flag)[i],
                        color=colors_closure[i], alpha=0.5)
            plt.plot(max_u_as_model, model_closure_full[i,:], 
                     color=colors_closure[i])
        plt.xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
        plt.ylabel('closure phase (deg)')
        if pdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = 'temp_pol%i_5.png' % idx
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        

        # VisPhi
        for i in range(0,6):
            plt.errorbar(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i], 
                        visphi_error[i,:]*(1-visphi_flag)[i],
                        color=colors_baseline[i], ls='', alpha=0.5, capsize=0)
            plt.scatter(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i],
                        color=colors_baseline[i], alpha=0.5)
            plt.plot(magu_as_model[i,:], model_visphi_full[i,:],
                    color=colors_baseline[i],alpha=1.0)
        plt.ylabel('visibility phase')
        plt.xlabel('spatial frequency (1/arcsec)')
        if pdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = 'temp_pol%i_6.png' % idx
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        
        



            
