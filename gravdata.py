from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import scipy as sp
import emcee
import corner
from multiprocessing import Pool
from fpdf import FPDF
from PIL import Image
from scipy import signal
from scipy import optimize 
from scipy import interpolate
import mpmath
from matplotlib import gridspec
from pkg_resources import resource_filename
from numba import njit
from astropy.time import Time
from datetime import timedelta, datetime
import sys
import os 
from joblib import Parallel, delayed

try:
    from generalFunctions import *
    set_style('show')
except (NameError, ModuleNotFoundError):
    pass


def debug(string):
    print('*******************')
    print(string)
    print('*******************')



color1 = '#C02F1D'
color2 = '#348ABD'
color3 = '#F26D21'
color4 = '#7A68A6'

def convert_date(date):
    t = Time(date)
    t2 = Time('2000-01-01T12:00:00')
    date_decimal = (t.mjd - t2.mjd)/365.25+2000
    
    date = date.replace('T', ' ')
    date = date.split('.')[0]
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return date_decimal, date

#@njit
def mathfunc_real(values, dt):
    return np.trapz(np.real(values), dx=dt, axis=0)
   
#@njit
def mathfunc_imag(values, dt):
    return np.trapz(np.imag(values), dx=dt, axis=0)

def complex_quadrature_num(func, a, b, theta, nsteps=int(1e2)):
    t = np.logspace(np.log10(a), np.log10(b), nsteps)
    dt = np.diff(t, axis=0)
    values = func(t, *theta)
    real_integral = mathfunc_real(values, dt)
    imag_integral = mathfunc_imag(values, dt)
    return real_integral + 1j*imag_integral


def procrustes(a,target,padval=0):
    try:
        if len(target) != a.ndim:
            raise TypeError('Target shape must have the same number of dimensions as the input')
    except TypeError:
        raise TypeError('Target must be array-like')

    try:
        #Get array in the right size to use
        b = np.ones(target,a.dtype)*padval
    except TypeError:
        raise TypeError('Pad value must be numeric')
    except ValueError:
        raise ValueError('Pad value must be scalar')

    aind = [slice(None,None)]*a.ndim
    bind = [slice(None,None)]*a.ndim

    for dd in range(a.ndim):
        if a.shape[dd] > target[dd]:
            diff = (a.shape[dd]-target[dd])/2.
            aind[dd] = slice(int(np.floor(diff)),int(a.shape[dd]-np.ceil(diff)))
        elif a.shape[dd] < target[dd]:
            diff = (target[dd]-a.shape[dd])/2.
            bind[dd] = slice(int(np.floor(diff)),int(target[dd]-np.ceil(diff)))
            
    b[bind] = a[aind]
    return b


class GravData():
    def __init__(self, data, verbose=True):
        self.name = data
        self.verbose = verbose
        self.colors_baseline = np.array(['k', 'darkblue', color4, 
                                         color2, 'darkred', color1])
        self.colors_closure = np.array([color1, 'darkred', 'k', color2])
        
        poscatg = ['VIS_DUAL_SCI_RAW', 'VIS_SINGLE_SCI_RAW', 'VIS_SINGLE_CAL_RAW', 
           'VIS_DUAL_CAL_RAW', 'VIS_SINGLE_CALIBRATED', 'VIS_DUAL_CALIBRATED',
           'SINGLE_SCI_VIS','SINGLE_SCI_VIS_CALIBRATED','DUAL_SCI_VIS', 
           'DUAL_SCI_VIS_CALIBRATED','SINGLE_CAL_VIS','DUAL_CAL_VIS', 'ASTROREDUCED',
           'DUAL_SCI_P2VMRED']
        
        header = fits.open(self.name)[0].header
        date = header['DATE-OBS']
        #if header['HIERARCH ESO INS OPTI11 ID'] == 'ONAXIS':
            #onaxis = True
            #print('Found onaxis data!')
        #else:
            #onaxis = False
            
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
        
        tel = fits.open(self.name)[0].header["TELESCOP"]
        if tel == 'ESO-VLTI-U1234':
            self.tel = 'UT'
        elif tel == 'ESO-VLTI-A1234':
            self.tel = 'AT'
        else:
            raise ValueError('Telescope not AT or UT, seomtehign wrong with input data')
        
        # Get BL names
        if self.raw:
            self.baseline_labels = np.array(["UT4-3","UT4-2","UT4-1",
                                            "UT3-2","UT3-1","UT2-1"])
            self.closure_labels = np.array(["UT4-3-2","UT4-3-1","UT4-2-1","UT3-2-1"])
        else:
            baseline_labels = []
            closure_labels = []
            tel_name = fits.open(self.name)['OI_ARRAY'].data['TEL_NAME']
            sta_index = fits.open(self.name)['OI_ARRAY'].data['STA_INDEX']
            if self.polmode == 'SPLIT':
                vis_index = fits.open(self.name)['OI_VIS',11].data['STA_INDEX']
                t3_index = fits.open(self.name)['OI_T3',11].data['STA_INDEX']
            else:
                vis_index = fits.open(self.name)['OI_VIS',10].data['STA_INDEX']
                t3_index = fits.open(self.name)['OI_T3',10].data['STA_INDEX']
            for bl in range(6):
                t1 = np.where(sta_index == vis_index[bl,0])[0][0]
                t2 = np.where(sta_index == vis_index[bl,1])[0][0]
                baseline_labels.append(tel_name[t1] + '-' + tel_name[t2][2])
            for cl in range(4):
                t1 = np.where(sta_index == t3_index[cl,0])[0][0]
                t2 = np.where(sta_index == t3_index[cl,1])[0][0]
                t3 = np.where(sta_index == t3_index[cl,2])[0][0]
                closure_labels.append(tel_name[t1] + '-' + tel_name[t2][2]+ '-' + tel_name[t3][2])
            self.closure_labels = np.array(closure_labels)
            self.baseline_labels = np.array(baseline_labels)
        
        if self.verbose:
            print('Data loaded as:')
            print('Telescope: ' + self.tel)
            print('Polarization: ' + self.polmode)
            print('Resolution: ' + self.resolution)
            print('Category: ' + self.datacatg)
        
        
        if not self.raw:
            if self.polmode == 'SPLIT':
                self.wlSC_P1 = fits.open(self.name)['OI_WAVELENGTH', 11].data['EFF_WAVE']*1e6
                self.wlSC_P2 = fits.open(self.name)['OI_WAVELENGTH', 12].data['EFF_WAVE']*1e6
                self.wlSC = self.wlSC_P1
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
        """
        Get data from one or optional two extensions
        """
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
        """
        Get the flux data
        """
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



    def getIntdata(self, mode='SC', plot=False, plotTAmp=False, flag=False):
        """
        Reads out all interferometric data and saves it into the class:
        visamp, visphi, visphi2, closure amplitude and phase
        if plot it plots all data
        
        """
        if self.raw:
            raise ValueError('Input is a RAW file, not usable for this function')
        
        fitsdata = fits.open(self.name)
        if self.polmode == 'SPLIT':
            if mode =='SC':
                self.u = fitsdata['OI_VIS', 11].data.field('UCOORD')
                self.v = fitsdata['OI_VIS', 11].data.field('VCOORD')
                
                # spatial frequency
                spFrequ = np.sqrt(self.u**2.+self.v**2.)
                wave = self.wlSC_P1
                self.wave = wave
                u_as = np.zeros((len(self.u),len(wave)))
                v_as = np.zeros((len(self.v),len(wave)))
                for i in range(0,len(self.u)):
                    u_as[i,:] = self.u[i]/(wave*1.e-6) * np.pi / 180. / 3600. # 1/as
                    v_as[i,:] = self.v[i]/(wave*1.e-6) * np.pi / 180. / 3600. # 1/as
                self.spFrequAS = np.sqrt(u_as**2.+v_as**2.)

                # spatial frequency T3
                magu = np.sqrt(self.u**2.+self.v**2.)
                max_spf = np.zeros((4))
                max_spf[0] = np.max(np.array([magu[0],magu[3],magu[1]]))
                max_spf[1] = np.max(np.array([magu[0],magu[4],magu[2]]))
                max_spf[2] = np.max(np.array([magu[1],magu[5],magu[2]]))
                max_spf[3] = np.max(np.array([magu[3],magu[5],magu[4]]))
                self.max_spf = max_spf
                spFrequAS_T3 = np.zeros((len(max_spf),len(wave)))
                for idx in range(4):
                    spFrequAS_T3[idx] = max_spf[idx]/(wave*1.e-6) * np.pi / 180. / 3600. # 1/as
                self.spFrequAS_T3 = spFrequAS_T3
                self.bispec_ind = np.array([[0,3,1],
                                            [0,4,2],
                                            [1,5,2],
                                            [3,5,4]])
                # Data
                # P1
                self.visampSC_P1 = fitsdata['OI_VIS', 11].data.field('VISAMP')
                self.visamperrSC_P1 = fitsdata['OI_VIS', 11].data.field('VISAMPERR')
                self.visphiSC_P1 = fitsdata['OI_VIS', 11].data.field('VISPHI')
                self.visphierrSC_P1 = fitsdata['OI_VIS', 11].data.field('VISPHIERR')
                self.vis2SC_P1 = fitsdata['OI_VIS2', 11].data.field('VIS2DATA')
                self.vis2errSC_P1 = fitsdata['OI_VIS2', 11].data.field('VIS2ERR')
                self.t3SC_P1 = fitsdata['OI_T3', 11].data.field('T3PHI')
                self.t3errSC_P1 = fitsdata['OI_T3', 11].data.field('T3PHIERR')
                self.t3ampSC_P1 = fitsdata['OI_T3', 11].data.field('T3AMP')
                self.t3amperrSC_P1 = fitsdata['OI_T3', 11].data.field('T3AMPERR')
                
                # P2
                self.visampSC_P2 = fitsdata['OI_VIS', 12].data.field('VISAMP')
                self.visamperrSC_P2 = fitsdata['OI_VIS', 12].data.field('VISAMPERR')
                self.visphiSC_P2 = fitsdata['OI_VIS', 12].data.field('VISPHI')
                self.visphierrSC_P2 = fitsdata['OI_VIS', 12].data.field('VISPHIERR')
                self.vis2SC_P2 = fitsdata['OI_VIS2', 12].data.field('VIS2DATA')
                self.vis2errSC_P2 = fitsdata['OI_VIS2', 12].data.field('VIS2ERR')
                self.t3SC_P2 = fitsdata['OI_T3', 12].data.field('T3PHI')
                self.t3errSC_P2 = fitsdata['OI_T3', 12].data.field('T3PHIERR')
                self.t3ampSC_P2 = fitsdata['OI_T3', 12].data.field('T3AMP')
                self.t3amperrSC_P2 = fitsdata['OI_T3', 12].data.field('T3AMPERR')
                
                # Flags
                self.visampflagSC_P1 = fitsdata['OI_VIS', 11].data.field('FLAG')
                self.visampflagSC_P2 = fitsdata['OI_VIS', 12].data.field('FLAG')
                self.vis2flagSC_P1 = fitsdata['OI_VIS2', 11].data.field('FLAG')
                self.vis2flagSC_P2 = fitsdata['OI_VIS2', 12].data.field('FLAG')
                self.t3flagSC_P1 = fitsdata['OI_T3', 11].data.field('FLAG')
                self.t3flagSC_P2 = fitsdata['OI_T3', 12].data.field('FLAG')
                self.t3ampflagSC_P1 = fitsdata['OI_T3', 11].data.field('FLAG')
                self.t3ampflagSC_P2 = fitsdata['OI_T3', 12].data.field('FLAG')
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

                    self.t3ampSC_P1[self.t3ampflagSC_P1] = np.nan
                    self.t3amperrSC_P1[self.t3ampflagSC_P1] = np.nan
                    self.t3ampSC_P2[self.t3ampflagSC_P2] = np.nan
                    self.t3amperrSC_P2[self.t3ampflagSC_P2] = np.nan

                    self.visphiSC_P1[self.visphiflagSC_P1] = np.nan
                    self.visphierrSC_P1[self.visphiflagSC_P1] = np.nan
                    self.visphiSC_P2[self.visphiflagSC_P2] = np.nan
                    self.visphierrSC_P2[self.visphiflagSC_P2] = np.nan

                if plot:
                    if plotTAmp:
                        gs = gridspec.GridSpec(3,2)
                        plt.figure(figsize=(15,15))
                    else:
                        gs = gridspec.GridSpec(2,2)
                        plt.figure(figsize=(15,12))
                        
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

                    axis = plt.subplot(gs[1,0])
                    for idx in range(len(self.t3SC_P2)):
                        plt.errorbar(self.spFrequAS_T3[idx,:], self.t3SC_P2[idx,:], self.t3errSC_P2[idx,:], marker='o',color=self.colors_closure[idx%4],linestyle='')
                    for idx in range(len(self.t3SC_P2)):
                        plt.errorbar(self.spFrequAS_T3[idx,:], self.t3SC_P1[idx,:], self.t3errSC_P1[idx,:], marker='p',color=self.colors_closure[idx%4],linestyle='')
                    plt.axhline(0, ls='--', lw=0.5)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('closure phase (deg)')
                    
                    axis = plt.subplot(gs[1,1])
                    if plotTAmp:
                        for idx in range(len(self.t3SC_P2)):
                            plt.errorbar(self.spFrequAS_T3[idx,:], self.t3ampSC_P2[idx,:], self.t3amperrSC_P2[idx,:], marker='o',color=self.colors_closure[idx%4],linestyle='')
                        for idx in range(len(self.t3SC_P2)):
                            plt.errorbar(self.spFrequAS_T3[idx,:], self.t3ampSC_P1[idx,:], self.t3amperrSC_P1[idx,:], marker='p',color=self.colors_closure[idx%4],linestyle='')
                        plt.axhline(1, ls='--', lw=0.5)
                        plt.ylim(-0.0,1.1)
                        plt.xlabel('spatial frequency (1/arcsec)')
                        plt.ylabel('closure amplitude')

                        axis = plt.subplot(gs[2,1])
                        
                    for idx in range(len(self.vis2SC_P1)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visphiSC_P1[idx,:],self.visphierrSC_P1[idx,:], ls='', marker='o',color=self.colors_baseline[idx%6])
                    for idx in range(len(self.vis2SC_P2)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visphiSC_P2[idx,:],self.visphierrSC_P2[idx,:], ls='', marker='p',color=self.colors_baseline[idx%6])
                    plt.axhline(0, ls='--', lw=0.5)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('visibility phase')

                    plt.show()
        if self.polmode =='COMBINED':
            if mode =='SC':
                self.u = fitsdata['OI_VIS', 10].data.field('UCOORD')
                self.v = fitsdata['OI_VIS', 10].data.field('VCOORD')
                
                # spatial frequency
                spFrequ = np.sqrt(self.u**2.+self.v**2.)
                wave = self.wlSC
                self.wave = wave
                u_as = np.zeros((len(self.u),len(wave)))
                v_as = np.zeros((len(self.v),len(wave)))
                for i in range(0,len(self.u)):
                    u_as[i,:] = self.u[i]/(wave*1.e-6) * np.pi / 180. / 3600. # 1/as
                    v_as[i,:] = self.v[i]/(wave*1.e-6) * np.pi / 180. / 3600. # 1/as
                self.spFrequAS = np.sqrt(u_as**2.+v_as**2.)

                # spatial frequency T3
                magu = np.sqrt(self.u**2.+self.v**2.)
                max_spf = np.zeros((4))
                max_spf[0] = np.max(np.array([magu[0],magu[3],magu[1]]))
                max_spf[1] = np.max(np.array([magu[0],magu[4],magu[2]]))
                max_spf[2] = np.max(np.array([magu[1],magu[5],magu[2]]))
                max_spf[3] = np.max(np.array([magu[3],magu[5],magu[4]]))
                self.max_spf = max_spf
                spFrequAS_T3 = np.zeros((len(max_spf),len(wave)))
                for idx in range(4):
                    spFrequAS_T3[idx] = max_spf[idx]/(wave*1.e-6) * np.pi / 180. / 3600. # 1/as
                self.spFrequAS_T3 = spFrequAS_T3
                self.bispec_ind = np.array([[0,3,1],
                                            [0,4,2],
                                            [1,5,2],
                                            [3,5,4]])
                # Data
                # P1
                self.visampSC = fitsdata['OI_VIS', 10].data.field('VISAMP')
                self.visamperrSC = fitsdata['OI_VIS', 10].data.field('VISAMPERR')
                self.visphiSC = fitsdata['OI_VIS', 10].data.field('VISPHI')
                self.visphierrSC = fitsdata['OI_VIS', 10].data.field('VISPHIERR')
                self.vis2SC = fitsdata['OI_VIS2', 10].data.field('VIS2DATA')
                self.vis2errSC = fitsdata['OI_VIS2', 10].data.field('VIS2ERR')
                self.t3SC = fitsdata['OI_T3', 10].data.field('T3PHI')
                self.t3errSC = fitsdata['OI_T3', 10].data.field('T3PHIERR')
                self.t3ampSC = fitsdata['OI_T3', 10].data.field('T3AMP')
                self.t3amperrSC = fitsdata['OI_T3', 10].data.field('T3AMPERR')
                
                # Flags
                self.visampflagSC = fitsdata['OI_VIS', 10].data.field('FLAG')
                self.vis2flagSC = fitsdata['OI_VIS2', 10].data.field('FLAG')
                self.t3flagSC = fitsdata['OI_T3', 10].data.field('FLAG')
                self.t3ampflagSC = fitsdata['OI_T3', 10].data.field('FLAG')
                self.visphiflagSC = fitsdata['OI_VIS', 10].data.field('FLAG')
                
                if flag:
                    self.visampSC[self.visampflagSC] = np.nan
                    self.visamperrSC[self.visampflagSC] = np.nan
                    
                    self.vis2SC[self.vis2flagSC] = np.nan
                    self.vis2errSC[self.vis2flagSC] = np.nan
                    
                    self.t3SC[self.t3flagSC] = np.nan
                    self.t3errSC[self.t3flagSC] = np.nan

                    self.t3ampSC[self.t3ampflagSC] = np.nan
                    self.t3amperrSC[self.t3ampflagSC] = np.nan

                    self.visphiSC[self.visphiflagSC] = np.nan
                    self.visphierrSC[self.visphiflagSC] = np.nan

                if plot:
                    if plotTAmp:
                        gs = gridspec.GridSpec(3,2)
                        plt.figure(figsize=(15,15))
                    else:
                        gs = gridspec.GridSpec(2,2)
                        plt.figure(figsize=(15,12))
                    axis = plt.subplot(gs[0,0])
                    for idx in range(len(self.vis2SC)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visampSC[idx,:], self.visamperrSC[idx,:], ls='', marker='o',color=self.colors_baseline[idx%6])
                    plt.axhline(1, ls='--', lw=0.5)
                    plt.ylim(-0.0,1.1)
                    plt.ylabel('visibility amplitude')

                    axis = plt.subplot(gs[0,1])
                    for idx in range(len(self.vis2SC)):
                        plt.errorbar(self.spFrequAS[idx,:], self.vis2SC[idx,:], self.vis2errSC[idx,:], ls='', marker='o',color=self.colors_baseline[idx%6])
                    plt.axhline(1, ls='--', lw=0.5)
                    plt.ylim(-0.0,1.1)
                    plt.ylabel('visibility squared')

                    axis = plt.subplot(gs[1,0])
                    for idx in range(len(self.t3SC)):
                        plt.errorbar(self.spFrequAS_T3[idx,:], self.t3SC[idx,:], self.t3errSC[idx,:], marker='o',color=self.colors_closure[idx%4],linestyle='')
                    plt.axhline(0, ls='--', lw=0.5)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('closure phase (deg)')
                    
                    axis = plt.subplot(gs[1,1])
                    if plotTAmp:
                        for idx in range(len(self.t3SC)):
                            plt.errorbar(self.spFrequAS_T3[idx,:], self.t3ampSC[idx,:], self.t3amperrSC[idx,:], marker='o',color=self.colors_closure[idx%4],linestyle='')
                        plt.axhline(1, ls='--', lw=0.5)
                        plt.ylim(-0.0,1.1)
                        plt.xlabel('spatial frequency (1/arcsec)')
                        plt.ylabel('closure amplitude')
                    
                        axis = plt.subplot(gs[2,1])
                    for idx in range(len(self.vis2SC)):
                        plt.errorbar(self.spFrequAS[idx,:], self.visphiSC[idx,:],self.visphierrSC[idx,:], ls='', marker='o',color=self.colors_baseline[idx%6])
                    plt.axhline(0, ls='--', lw=0.5)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('visibility phase')

                    plt.show()
        fitsdata.close()
                
                
    def getFluxfromRAW(self, flatfile, method, skyfile=None, wavefile=None, 
                       pp_wl=None, flatflux=False):
        """
        Get the flux values from a raw file
        method has to be 'spectrum', 'preproc', 'p2vmred', 'dualscivis'
        Depending on the method the flux extraction from the raw detector
        frames is done until the given endproduct
        """
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
                    red_spectra_i[tdx,idx,:] = interpolate.interp1d(wavefits['WAVE_DATA_SC'].data['DATA%i' % (idx+1)][0],
                                                                    red_spectra[tdx,idx,:])(pp_wl)
                except ValueError:
                    red_spectra_i[tdx,idx,:] = interpolate.interp1d(wavefits['WAVE_DATA_SC'].data['DATA%i' % (idx+1)][0],
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
    
    
    
    def getDlambda(self, idel=False):
        """
        Get the size of the spectral channels
        if idel it is taken from the hardcoded size of the response functions,
        otherwise it is read in from the OI_WAVELENGTH table
        TODO Idel + medium/high resolution needs some work
        """
        if idel:
            nwave = self.channel
            wave = self.wlSC
            self.wave = wave
            if nwave in [11, 14]:
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
                elif self.polmode == 'SPLIT':
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
        else:
            nwave = self.channel
            if self.polmode == 'COMBINED':
                effband = fits.open(self.name)['OI_WAVELENGTH', 10].data['EFF_BAND']
            elif self.polmode == 'SPLIT':
                effband = fits.open(self.name)['OI_WAVELENGTH', 11].data['EFF_BAND']
            dlambda = np.zeros((6,nwave))
            for idx in range(6):
                dlambda[idx] = effband/2*1e6
        self.dlambda = dlambda 
    
       
    
    
    
    ############################################    
    ############################################
    ################ Phase Maps ################
    ############################################
    ############################################
    
    def createPhasemaps(self, nthreads=1, smooth=15, zerfile='phasemap_zernike_20200629.npy'):
        
        def phase_screen(A00, A1m1, A1p1, A2m2, A2p2, A20, A3m1, A3p1, A3m3, A3p3, 
                         A4m2, A4p2, A4m4, A4p4, A40, d=8,
                         MFR=0.6308, stopB=8.0, stopS=0.96, 
                         dalpha=1, totN=6e3, lam0=2.2, fit=True):
            
            """
            Simulate phase screens taking into account static aberrations.
            
            Parameters:
            -----------
            * Static aberrations in the pupil plane are described by low-order Zernicke polynomials
            * Their amplitudes are in units of micro-meter

            A00  (float) : piston
            A1m1 (float) : vertical tilt
            A1p1 (float) : horizontal tilt
            A2m2 (float) : vertical astigmatism
            A2p2 (float) : horizontal astigmatism
            A20  (float) : defocuss
            A3m1 (float) : vertical coma
            A3p1 (float) : horizontal coma
            A3m3 (float) : vertical trefoil
            A3p3 (float) : oblique trefoil
            A4m2 (float) : oblique secondary astigmatism
            A4p2 (float) : vertical secondary astigmatism
            A4m4 (float) : oblique quadrafoil
            A4p4 (float) : vertical quadrafoil
            A40  (float) : primary spherical

            * optical system
            MFR (float)   : sigma of the fiber mode profile in units of dish radius
            stopB (float) : outer stop diameter in meters
            stopS (float) : inner stop diameter in meters
            -> for measurments with the GRAVITY GCU the dish diameter and central obscuration do not determine the stop size

            * further parameters specify the output grid
            dalpha (float) : pixel width in the imaging plane in mas
            totN   (float) : total number of pixels in the pupil plane
            lam0   (float) : wavelength at which the phase screen is computed in micro-meter
            
            """ 
            
            #--- instrument parameters ---#
            d1   = d      # dish diameter in meters
            
            #--- coordinate scaling ---#
            lam0   = lam0*1e-6
            mas    = 1.e-3 * (2.*np.pi/360) *1./3600
            ext    = totN*d1/lam0*mas*dalpha*dalpha
            du     = dalpha/ext*d1/lam0
            dishN  = 2*ext/dalpha    
                
            #--- coordinates ---#
            ii     = np.arange(totN) - (totN/2)
            ii     = np.fft.fftshift(ii)
            
            # image plane
            a1, a2 = np.meshgrid(ii*dalpha, ii*dalpha)
            aa     = np.sqrt(a1*a1 + a2*a2)
            
            # pupil plane
            u1, u2 = np.meshgrid(ii*du, ii*du)
            uu     = np.sqrt( u1*u1 + u2*u2 )
            r      = lam0*uu
            t      = np.angle(u1 + 1j*u2)
            
            # cut out slice
            cn   = int(dishN/2)
            ss   = slice(int(totN//2)-cn, int(totN//2)+cn+1)
            amax = dalpha*cn
            
            #--- pupil function ---#
            pupil = np.logical_and( r<(stopB/2.), r>(stopS/2.) )
            
            #--- fiber profile ---#
            fiber = np.exp(-0.5*(r/(MFR*d1/2.))**2)
            
            #--- phase screens (pupil plane) ---#
            zernike  = A00
            zernike += A1m1*2*(2.*r/d1)*np.sin(t)
            zernike += A1p1*2*(2.*r/d1)*np.cos(t)
            zernike += A2m2*np.sqrt(6.)*(2.*r/d1)**2*np.sin(2.*t)
            zernike += A2p2*np.sqrt(6.)*(2.*r/d1)**2*np.cos(2.*t)
            zernike += A20 *np.sqrt(3.)*(2.*(2.*r/d1)**2 - 1)
            zernike += A3m1*np.sqrt(8.)*(3.*(2.*r/d1)**3 - 2.*(2.*r/d1))*np.sin(t)
            zernike += A3p1*np.sqrt(8.)*(3.*(2.*r/d1)**3 - 2.*(2.*r/d1))*np.cos(t)
            zernike += A3m3*np.sqrt(8.)*(2.*r/d1)**3*np.sin(3.*t)
            zernike += A3p3*np.sqrt(8.)*(2.*r/d1)**3*np.cos(3.*t)
            zernike += A4m2*np.sqrt(10.)*(2.*r/d1)**4*np.sin(4.*t)
            zernike += A4p2*np.sqrt(10.)*(2.*r/d1)**4*np.cos(4.*t)
            zernike += A4m4*np.sqrt(10.)*(4.*(2.*r/d1)**4 -3.*(2.*r/d1)**2)*np.sin(2.*t)
            zernike += A4p4*np.sqrt(10.)*(4.*(2.*r/d1)**4 -3.*(2.*r/d1)**2)*np.cos(2.*t)
            zernike += A40*np.sqrt(5.)*(6.*(2.*r/d1)**4 - 6.*(2.*r/d1)**2 + 1)
            
            phase = 2.*np.pi/lam0*zernike*1.e-6

            #--- transform to image plane ---#
            complexPsf = np.fft.fftshift(np.fft.fft2( pupil * fiber * np.exp(1j*phase) ))
            if fit:
                return complexPsf[ss,ss]/np.abs(complexPsf[ss,ss]).max()
            else:
                return complexPsf, amax, ss, 2*cn+1
        
        zernikefile = resource_filename('gravipy', zerfile)
        zer = np.load(zernikefile, allow_pickle=True).item()

        wave = self.wlSC
        
        if self.tel == 'UT':
            stopB=8.0
            stopS=0.96
            dalpha=1
            totN=6e3
            d = 8
            
        elif self.tel == 'AT':
            stopB = 8.0/4.4
            stopS = 8.0/4.4*0.076
            dalpha = 1.*4.4
            totN = 6e3
            d = 1.8
        
        kernel = Gaussian2DKernel(x_stddev=smooth)
         
        print('Creating phasemaps:')
        if nthreads == 1:
            all_pm = np.zeros((len(wave), 4, 201, 201),
                            dtype=np.complex_)
            all_pm_denom = np.zeros((len(wave), 4, 201, 201),
                                    dtype=np.complex_)
            for wdx, wl in enumerate(wave):
                print_status(wdx, len(wave))
                for GV in range(4):
                    zer_GV = zer['GV%i' % (GV+1)]
                    pm = phase_screen(*zer_GV, lam0=wl, d=d, stopB=stopB, stopS=stopS, 
                                      dalpha=dalpha, totN=totN)
                    pm = procrustes(pm, (201,201), padval=0)
                    pm_sm = signal.convolve2d(pm, kernel, mode='same')
                    pm_sm_denom = signal.convolve2d(np.abs(pm)**2, kernel, mode='same')
                    
                    all_pm[wdx, GV] = pm_sm
                    all_pm_denom[wdx, GV] = pm_sm_denom
        else:
            def multi_pm(lam):
                m_all_pm = np.zeros((4, 201, 201), dtype=np.complex_)
                m_all_pm_denom = np.zeros((4, 201, 201), dtype=np.complex_)
                for GV in range(4):
                    zer_GV = zer['GV%i' % (GV+1)]
                    pm = phase_screen(*zer_GV, lam0=lam, d=d, stopB=stopB, stopS=stopS, 
                                      dalpha=dalpha, totN=totN)
                    pm = procrustes(pm, (201,201), padval=0)
                    pm_sm = signal.convolve2d(pm, kernel, mode='same')
                    pm_sm_denom = signal.convolve2d(np.abs(pm)**2, kernel, mode='same')
                    m_all_pm[GV] = pm_sm
                    m_all_pm_denom[GV] = pm_sm_denom
                return np.array([m_all_pm, m_all_pm_denom])
            
            res = np.array(Parallel(n_jobs=nthreads)(delayed(multi_pm)(lam) for lam in wave))
            
            all_pm = res[:,0,:,:,:]
            all_pm_denom = res[:,1,:,:,:]
                
        savename = 'Phasemap_%s_%s_Smooth%i.npy' % (self.tel, self.resolution, smooth)
        savefile = resource_filename('gravipy', savename)
        np.save(savefile, all_pm)
        savename = 'Phasemap_%s_%s_Smooth%i_denom.npy' % (self.tel, self.resolution, smooth)
        savefile = resource_filename('gravipy', savename)
        np.save(savefile, all_pm_denom)


    def rotation(self, ang):
        """
        Rotation matrix, needed for phasemaps
        """
        return np.array([[np.cos(ang), np.sin(ang)],
                         [-np.sin(ang), np.cos(ang)]])
    
    
    def loadPhasemaps(self, interp):
        if self.tel == 'UT':
            if self.resolution == 'LOW':
                pm1_file = 'Phasemap_UT_LOW_Smooth15.npy'
                pm2_file = 'Phasemap_UT_LOW_Smooth15_denom.npy'
                pm1 = np.load(resource_filename('gravipy', pm1_file))
                pm2 = np.real(np.load(resource_filename('gravipy', pm2_file)))
            else:
                raise ValueError('Have to create phasemaps first')
        elif self.tel == 'AT':
            if self.resolution == 'MEDIUM':
                pm1_file = 'Phasemap_AT_MEDIUM_Smooth18.npy'
                pm2_file = 'Phasemap_AT_MEDIUM_Smooth18_denom.npy'
                pm1 = np.load(resource_filename('gravipy', pm1_file))
                pm2 = np.real(np.load(resource_filename('gravipy', pm2_file)))
            else:
                raise ValueError('Have to create phasemaps first')

        wave = self.wlSC
        if pm1.shape[0] != len(wave):
            raise ValueError('Phasemap and data have different numbers of channels')

        self.amp_map = np.abs(pm1)
        self.pha_map = np.angle(pm1, deg=True)
        self.amp_map_denom = pm2

        for wdx in range(len(wave)):
            for tel in range(4):
                self.amp_map[wdx,tel] /= np.max(self.amp_map[wdx,tel])
                self.amp_map_denom[wdx,tel] /= np.max(self.amp_map_denom[wdx,tel])
                
        if interp:
            x = np.arange(201)
            y = np.arange(201)
            self.amp_map_int = np.zeros((len(wave),4), dtype=object)
            self.pha_map_int = np.zeros((len(wave),4), dtype=object)
            self.amp_map_denom_int = np.zeros((len(wave),4), dtype=object)
            for tel in range(4):
                for wdx in range(len(wave)):
                    self.amp_map_int[wdx, tel] = interpolate.interp2d(x, y, self.amp_map[wdx, tel])
                    self.pha_map_int[wdx, tel] = interpolate.interp2d(x, y, self.pha_map[wdx, tel])
                    self.amp_map_denom_int[wdx, tel] = interpolate.interp2d(x, y, self.amp_map_denom[wdx, tel])
                
        
        
    def readPhasemaps(self, ra, dec, fromFits=True, 
                      northangle=None, dra=None, ddec=None,
                      interp=True, givepos=False):
        """
        Calculates coupling amplitude / phase for given coordinates
        ra,dec: RA, DEC position on sky relative to nominal field center = SOBJ [mas]
        dra,ddec: ESO QC MET SOBJ DRA / DDEC: 
            location of science object (= desired science fiber position, = field center) 
            given by INS.SOBJ relative to *actual* fiber position measured by the laser metrology [mas]
            mis-pointing = actual - desired fiber position = -(DRA,DDEC)
        north_angle: north direction on acqcam in degree
        if fromFits is true, northangle & dra,ddec are taken from fits file
        """
        if fromFits:
            # should not do that in here for mcmc
            header = fits.open(self.name)[0].header
            northangle1 = header['ESO QC ACQ FIELD1 NORTH_ANGLE']/180*math.pi
            northangle2 = header['ESO QC ACQ FIELD2 NORTH_ANGLE']/180*math.pi
            northangle3 = header['ESO QC ACQ FIELD3 NORTH_ANGLE']/180*math.pi
            northangle4 = header['ESO QC ACQ FIELD4 NORTH_ANGLE']/180*math.pi
            northangle = [northangle1, northangle2, northangle3, northangle4]

            ddec1 = header['ESO QC MET SOBJ DDEC1']
            ddec2 = header['ESO QC MET SOBJ DDEC2']
            ddec3 = header['ESO QC MET SOBJ DDEC3']
            ddec4 = header['ESO QC MET SOBJ DDEC4']
            ddec = [ddec1, ddec2, ddec3, ddec4]

            dra1 = header['ESO QC MET SOBJ DRA1']
            dra2 = header['ESO QC MET SOBJ DRA2']
            dra3 = header['ESO QC MET SOBJ DRA3']
            dra4 = header['ESO QC MET SOBJ DRA4']
            dra = [dra1, dra2, dra3, dra4]
            
        wave = self.wlSC
        
        if self.simulate_pm:
            dra = np.zeros_like(np.array(dra))
            ddec = np.zeros_like(np.array(ddec))
            northangle = np.zeros_like(np.array(northangle))
            
        pm_pos = np.zeros((4, 2))
        cor_amp = np.zeros((4, len(wave)))
        cor_pha = np.zeros((4, len(wave)))
        cor_int_denom = np.zeros((4, len(wave)))
        
        for tel in range(4):
            pos = np.array([ra + dra[tel], dec + ddec[tel]])
            if self.tel == 'AT':
                pos /= 4.4
            pos_rot = np.dot(self.rotation(northangle[tel]), pos) + 100
            pm_pos[tel] = pos_rot
            
            
            for wdx in range(len(wave)):
                if interp:
                    cor_amp[tel, wdx] = self.amp_map_int[wdx, tel](pos_rot[0], pos_rot[1])
                    cor_pha[tel, wdx] = self.pha_map_int[wdx, tel](pos_rot[0], pos_rot[1])
                    cor_int_denom[tel, wdx] = self.amp_map_denom_int[wdx, tel](pos_rot[0], pos_rot[1])
                else:
                    pos_int = np.round(pos_rot).astype(int)
                    cor_amp[tel, wdx] = self.amp_map[wdx,tel][pos_int[0],pos_int[1]]
                    cor_pha[tel, wdx] = self.pha_map[wdx,tel][pos_int[0],pos_int[1]]
                    cor_int_denom[tel, wdx] = self.amp_map_denom[wdx,tel][pos_int[0],pos_int[1]]
        if givepos:
            return pm_pos
        else:
            return cor_amp, cor_pha, cor_int_denom 
        
        
                

    
    #def givePhasemapPos(self, ra, dec, wave, fromFits=True, 
                        #northangle=None, dra=None, ddec=None,):
        #"""
        #"""
        #if self.tel == 'UT':
            #scale = 1
        #elif self.tel == 'AT':
            #scale = 4.4
                
        #if fromFits:
            ## should not do that in here for mcmc
            #header = fits.open(self.name)[0].header
            #northangle1 = header['ESO QC ACQ FIELD1 NORTH_ANGLE']/180*math.pi
            #northangle2 = header['ESO QC ACQ FIELD2 NORTH_ANGLE']/180*math.pi
            #northangle3 = header['ESO QC ACQ FIELD3 NORTH_ANGLE']/180*math.pi
            #northangle4 = header['ESO QC ACQ FIELD4 NORTH_ANGLE']/180*math.pi
            #northangle = [northangle1, northangle2, northangle3, northangle4]

            #ddec1 = header['ESO QC MET SOBJ DDEC1']
            #ddec2 = header['ESO QC MET SOBJ DDEC2']
            #ddec3 = header['ESO QC MET SOBJ DDEC3']
            #ddec4 = header['ESO QC MET SOBJ DDEC4']
            #ddec = [ddec1, ddec2, ddec3, ddec4]

            #dra1 = header['ESO QC MET SOBJ DRA1']
            #dra2 = header['ESO QC MET SOBJ DRA2']
            #dra3 = header['ESO QC MET SOBJ DRA3']
            #dra4 = header['ESO QC MET SOBJ DRA4']
            #dra = [dra1, dra2, dra3, dra4]
            
        #lambda0 = 2.2
        #position = np.zeros((4, len(wave), 2))
        #for tel in range(4):
            #pos = np.array([ra + dra[tel], dec + ddec[tel]])
            #pos_rot = np.dot(self.rotation(northangle[tel]), pos)
            #for channel in range(len(wave)):
                #pos_scaled = pos_rot*lambda0/wave[channel]/scale + 100
                #pos_int = (np.round(pos_scaled)).astype(int)
                #position[tel, channel] = pos_scaled

        #return position
    
    
    
    
    def readPhasemapsSingle(self, ra, dec, northangle, dra, ddec, tel, lam, interp=True):
        """
        Same as readPhasemaps, but only for a single point, for testing
        """
        try:
           self.amp_map
        except AttributeError:
            self.loadPhasemaps(interp=True)

        pos = np.array([ra + dra, dec + ddec])
        if self.tel == 'AT':
            pos /= 4.4

        pos_rot = np.dot(self.rotation(northangle), pos)+100

        wave = self.wlSC
        wdx = find_nearest(wave, lam)
        
        print('Wavelength: %.2f given, will ues %.2f' % (lam, wave[wdx]))
        
        if interp:
            cor_amp = self.amp_map_int[wdx, tel](pos_rot[0], pos_rot[1])
            cor_pha = self.pha_map_int[wdx, tel](pos_rot[0], pos_rot[1])
            cor_int_denom = self.amp_map_denom_int[wdx, tel](pos_rot[0], pos_rot[1])
        else:
            pos_int = np.round(pos_rot).astype(int)
            cor_amp = self.amp_map[wdx,tel][pos_int[0],pos_int[1]]
            cor_pha = self.pha_map[wdx,tel][pos_int[0],pos_int[1]]
            cor_int_denom = self.amp_map_denom[wdx,tel][pos_int[0],pos_int[1]]
        print(cor_amp, cor_pha)
        return cor_amp, cor_pha, cor_int_denom
        
    
    
 
    
    ############################################    
    ############################################
    ############### Binary model ###############
    ############################################
    ############################################

    def vis_intensity_approx(self, s, alpha, lambda0, dlambda):
        """
        Approximation for Modulated interferometric intensity
        s:      B*skypos-opd1-opd2
        alpha:  power law index
        lambda0:zentral wavelength
        dlambda:size of channels 
        """
        x = 2*s*dlambda/lambda0**2.
        sinc = np.sinc(x/np.pi)
        return (lambda0/2.2)**(-1-alpha)*2*dlambda*sinc*np.exp(-2.j*np.pi*s/lambda0)
    
    def vis_intensity(self, s, alpha, lambda0, dlambda):
        """
        Analytic solution for Modulated interferometric intensity
        s:      B*skypos-opd1-opd2
        alpha:  power law index
        lambda0:zentral wavelength
        dlambda:size of channels 
        """
        x1 = lambda0+dlambda
        x2 = lambda0-dlambda
        if not np.isscalar(lambda0):
            if not np.isscalar(s):
                res = np.zeros(len(lambda0), dtype=np.complex_)
                for idx in range(len(lambda0)):
                    if s[idx] == 0 and alpha == 0:
                        res[idx] = self.vis_intensity_num(s[idx], alpha, 
                                                          lambda0[idx], dlambda[idx])
                    else:
                        up = self.vis_int_full(s[idx], alpha, x1[idx])
                        low = self.vis_int_full(s[idx], alpha, x2[idx])
                        res[idx] = up - low
            else:
                res = np.zeros(len(lambda0), dtype=np.complex_)
                for idx in range(len(lambda0)):
                    if s == 0 and alpha == 0:
                        res[idx] = self.vis_intensity_num(s, alpha, lambda0[idx], 
                                                          dlambda[idx])
                    else:
                        up = self.vis_int_full(s, alpha, x1[idx])
                        low = self.vis_int_full(s, alpha, x2[idx])
                        res[idx] = up - low
        else:
            if s == 0 and alpha == 0:
                res = self.vis_intensity_num(s, alpha, lambda0, dlambda)
            else:
                up = self.vis_int_full(s, alpha, x1)
                low = self.vis_int_full(s, alpha, x2)
                res = up - low
        return res
        
    def vis_int_full(self, s, alpha, difflam):
        if s == 0:
            return -2.2**(1 + alpha)/alpha*difflam**(-alpha)
        a = difflam*(difflam/2.2)**(-1-alpha)
        bval = mpmath.gammainc(alpha, (2*1j*np.pi*s/difflam))
        b = float(bval.real)+float(bval.imag)*1j
        c = (2*np.pi*1j*s/difflam)**alpha
        return (a*b/c)
    
    
    def visibility_integrator(self, wave, s, alpha):
        """
        complex integral to be integrated over wavelength
        wave in [micron]
        theta holds the exponent alpha, and the seperation s
        """
        return (wave/2.2)**(-1-alpha)*np.exp(-2*np.pi*1j*s/wave)
    
    
    def vis_intensity_num(self, s, alpha, lambda0, dlambda):
        """
        Dull numeric solution for Modulated interferometric intensity
        s:      B*skypos-opd1-opd2
        alpha:  power law index
        lambda0:zentral wavelength
        dlambda:size of channels 
        """
        if np.all(s == 0.) and alpha != 0:
            return -2.2**(1 + alpha)/alpha*(lambda0+dlambda)**(-alpha) - (-2.2**(1 + alpha)/alpha*(lambda0-dlambda)**(-alpha))
        else:
            return complex_quadrature_num(self.visibility_integrator, lambda0-dlambda, lambda0+dlambda, (s, alpha))
    
    
    
    def simulateVisdata_single(self, theta, wave, dlambda, u, v, 
                               fixedBG=True, fixedBH=True, 
                               phasemaps=False, phasemapsstuff=None,
                               interppm=True, approx="approx"):
        '''
        Test function to generate a single datapoint for a given u, v, lambda, dlamba & theta
        
        Theta should be a list of:
        dRA, dDEC, f, alpha flare, f BG, (alpha BG), PC RA, PC DEC
        
        Values in bracket are by default not used, can be activated by options:
        fixedBH:        Keep primary power law [False]        
        fixedBG:        Keep background power law [True]  
        
        if phasemaps=True, phasemapsstuff must be a list of:
            [tel1, dra1, ddec1, north_angle1, tel2, dra2, ddec2, north_angle2]
        '''
        
        theta_names_raw = np.array(["dRA", "dDEC", "f", "alpha flare", "f BG", 
                                    "alpha BG", "PC RA", "PC DEC"])
        rad2as = 180 / np.pi * 3600
        # check Theta
        try:
            if len(theta) != 8:
                print('Theta has to include the following 8 parameter:')
                print(theta_names_raw)
                raise ValueError('Wrong number of input parameter given (should be 8)')
        except(TypeError):
            print('Thetha has to include the following 8 parameter:')
            print(theta_names_raw)
            raise ValueError('Wrong number of input parameter given (should be 8)') 
        
        mas2rad = 1e-3 / 3600 / 180 * np.pi
        dRA = theta[0]
        dDEC = theta[1]
        f = theta[2]
        if fixedBH:
            alpha_SgrA = -0.5
        else:
            alpha_SgrA = theta[3]
        fluxRatioBG = theta[4]
        if fixedBG:
            alpha_bg = 3.
        else:
            alpha_bg = theta[5]
        phaseCenterRA = theta[6]
        phaseCenterDEC = theta[7]
        alpha_S2 = 3
        f = 10.**f

        s_SgrA = ((phaseCenterRA)*u + (phaseCenterDEC)*v) * mas2rad * 1e6
        s_S2 = ((dRA+phaseCenterRA)*u + (dDEC+phaseCenterDEC)*v) * mas2rad * 1e6
                
        if phasemaps:
            (tel1, dra1, ddec1, north_angle1, 
             tel2, dra2, ddec2, north_angle2) = phasemapsstuff

            cor_amp_sgr1, cor_pha_sgr1, cor_int_sgr1 = self.readPhasemapsSingle(phaseCenterRA,
                                                                  phaseCenterDEC,
                                                                  north_angle1, 
                                                                  dra1, ddec1,
                                                                  tel1, wave,
                                                                  interp=interppm)
            cor_amp_sgr2, cor_pha_sgr2, cor_int_sgr2 = self.readPhasemapsSingle(phaseCenterRA,
                                                                  phaseCenterDEC,
                                                                  north_angle2, 
                                                                  dra2, ddec2,
                                                                  tel2, wave,
                                                                  interp=interppm)
            cor_amp_s21, cor_pha_s21, cor_int_s21 = self.readPhasemapsSingle(dRA+phaseCenterRA, 
                                                               dDEC+phaseCenterDEC,
                                                               north_angle1, 
                                                               dra1, ddec1,
                                                               tel1, wave,
                                                               interp=interppm)
            cor_amp_s22, cor_pha_s22, cor_int_s22 = self.readPhasemapsSingle(dRA+phaseCenterRA, 
                                                               dDEC+phaseCenterDEC,
                                                               north_angle2, 
                                                               dra2, ddec2,
                                                               tel2, wave,
                                                               interp=interppm)
            # differential opd
            opd_sgr = (cor_pha_sgr1-cor_pha_sgr2)/360*wave
            s_SgrA -= opd_sgr
            opd_s2 = (cor_pha_s21-cor_pha_s22)/360*wave
            s_S2 -= opd_s2
            
            # different coupling
            cr1 = (cor_amp_s21 / cor_amp_sgr1)**2
            cr2 = (cor_amp_s22 / cor_amp_sgr2)**2
            cr_denom1 = (cor_int_s21 / cor_int_sgr1)
            cr_denom2 = (cor_int_s22/ cor_int_sgr2)

            if approx == "approx":
                intSgrA = self.vis_intensity_approx(s_SgrA, alpha_SgrA, wave, dlambda)
                intS2 = self.vis_intensity_approx(s_S2, alpha_S2, wave, dlambda)
                intSgrA_center = self.vis_intensity_approx(0, alpha_SgrA, wave, dlambda)
                intS2_center = self.vis_intensity_approx(0, alpha_S2, wave, dlambda)
                intBG = self.vis_intensity_approx(0, alpha_bg, wave, dlambda)
            elif approx == "analytic":
                intSgrA = self.vis_intensity(s_SgrA, alpha_SgrA, wave, dlambda)
                intS2 = self.vis_intensity(s_S2, alpha_S2, wave, dlambda)
                intSgrA_center = self.vis_intensity(0, alpha_SgrA, wave, dlambda)
                intS2_center = self.vis_intensity(0, alpha_S2, wave, dlambda)
                intBG = self.vis_intensity(0, alpha_bg, wave, dlambda)
            elif approx == "numeric":
                intSgrA = self.vis_intensity_num(s_SgrA, alpha_SgrA, wave, dlambda)
                intS2 = self.vis_intensity_num(s_S2, alpha_S2, wave, dlambda)
                intSgrA_center = self.vis_intensity_num(0, alpha_SgrA, wave, dlambda)
                intS2_center = self.vis_intensity_num(0, alpha_S2, wave, dlambda)
                intBG = self.vis_intensity_num(0, alpha_bg, wave, dlambda)
            else:
                raise ValueError("approx needs to be in [approx, analytic, numeric]")
            
            vis = ((intSgrA + f*np.sqrt(cr1*cr2)*intS2)/
                   (np.sqrt(intSgrA_center + f*cr_denom1*intS2_center + fluxRatioBG * intBG)*
                    np.sqrt(intSgrA_center + f*cr_denom2*intS2_center + fluxRatioBG * intBG)))
            
        else:
            if approx == "approx":
                intSgrA = self.vis_intensity_approx(s_SgrA, alpha_SgrA, wave, dlambda)
                intS2 = self.vis_intensity_approx(s_S2, alpha_S2, wave, dlambda)
                intSgrA_center = self.vis_intensity_approx(0, alpha_SgrA, wave, dlambda)
                intS2_center = self.vis_intensity_approx(0, alpha_S2, wave, dlambda)
                intBG = self.vis_intensity_approx(0, alpha_bg, wave, dlambda)
            elif approx == "analytic":
                intSgrA = self.vis_intensity(s_SgrA, alpha_SgrA, wave, dlambda)
                intS2 = self.vis_intensity(s_S2, alpha_S2, wave, dlambda)
                intSgrA_center = self.vis_intensity(0, alpha_SgrA, wave, dlambda)
                intS2_center = self.vis_intensity(0, alpha_S2, wave, dlambda)
                intBG = self.vis_intensity(0, alpha_bg, wave, dlambda)
            elif approx == "numeric":
                intSgrA = self.vis_intensity_num(s_SgrA, alpha_SgrA, wave, dlambda)
                intS2 = self.vis_intensity_num(s_S2, alpha_S2, wave, dlambda)
                intSgrA_center = self.vis_intensity_num(0, alpha_SgrA, wave, dlambda)
                intS2_center = self.vis_intensity_num(0, alpha_S2, wave, dlambda)
                intBG = self.vis_intensity_num(0, alpha_bg, wave, dlambda)
            else:
                raise ValueError('apprix has to be approx, analytic or numeric')
                
            
            vis = ((intSgrA + f * intS2)/
                    (intSgrA_center + f * intS2_center + fluxRatioBG * intBG))
            
        return vis
    
    
    def simulateVisdata(self, theta, constant_f=True, use_opds=False, fixedBG=True, fixedBH=True, fiberOff=None, 
                        plot=True, phasemaps=False, phasemapsstuff=None, interppm=True,
                        approx='numeric'):
        """
        Test function to see how a given theta would look like
        Theta should be a list of:
        dRA, dDEC, f1, (f2), (f3), (f4), alpha flare, f BG, (alpha BG), 
        PC RA, PC DEC, (OPD1), (OPD2), (OPD3), (OPD4)
        
        Values in bracket are by default not used, can be activated by options:
        constant_f:     Constant coupling [True]
        use_opds:       Fit OPDs [False] 
        fixedBG:        Keep background power law [True]

        if phasemaps=True, phasemapsstuff must be a list of:
            [dra, ddec, north_angle]
        
        """
        theta_names_raw = np.array(["dRA", "dDEC", "f1", "f2", "f3", "f4" , "alpha flare",
                                    "f BG", "alpha BG", "PC RA", "PC DEC", "OPD1", "OPD2",
                                    "OPD3", "OPD4"])
        rad2as = 180 / np.pi * 3600
        try:
            if len(theta) != 15:
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
        self.fixedBH = fixedBH
        self.interppm = interppm
        self.approx = approx
        self.fixpos = False
        self.specialfit = False
        
        if fiberOff is None:
            self.fiberOffX = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFX"] 
            self.fiberOffY = -fits.open(self.name)[0].header["HIERARCH ESO INS SOBJ OFFY"] 
        else:
            self.fiberOffX = fiberOff[0]
            self.fiberOffY = fiberOff[1]
        print("fiber center: %.2f, %.2f (mas)" % (self.fiberOffX, self.fiberOffY))
        
        if phasemaps:
            self.dra = phasemapsstuff[0]*np.ones(4)
            self.ddec = phasemapsstuff[1]*np.ones(4)
            self.northangle = phasemapsstuff[2]*np.ones(4)
            self.phasemaps = True
        else:
            self.phasemaps = False

        nwave = self.channel
        if nwave != 14:
            raise ValueError('Only usable for 14 channels')
        self.getIntdata(plot=False, flag=False)
        u = self.u
        v = self.v
        wave = self.wlSC_P1
        
        self.getDlambda()
        dlambda = self.dlambda

        visamp, visphi, closure, closamp = self.calc_vis(theta, u, v, wave, dlambda)
        vis2 = visamp**2
        
        if plot:
            gs = gridspec.GridSpec(2,2)
            plt.figure(figsize=(25,25))
            
            if phasemaps:
                wave_model = wave
            else:
                wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
            dlambda_model = np.zeros((6,len(wave_model)))
            for i in range(0,6):
                dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
            (model_visamp_full, model_visphi_full, 
            model_closure_full, model_closamp_full)  = self.calc_vis(theta, u, v, wave_model, dlambda_model)
            model_vis2_full = model_visamp_full**2.

            magu_as = self.spFrequAS
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
            maxval = []
            for i in range(0,4):
                max_u_as_model = self.max_spf[i]/(wave_model*1.e-6) / rad2as
                plt.errorbar(self.spFrequAS_T3[i,:], closure[i,:],
                             color=self.colors_closure[i],ls='', marker='o')
                plt.plot(max_u_as_model, model_closure_full[i,:],
                         color=self.colors_closure[i])
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

        return visamp, vis2, visphi, closure, closamp


    def calc_vis(self, theta, u, v, wave, dlambda):
        mas2rad = 1e-3 / 3600 / 180 * np.pi
        rad2mas = 180 / np.pi * 3600 * 1e3
        constant_f = self.constant_f
        fixedBG = self.fixedBG
        use_opds = self.use_opds
        fiberOffX = self.fiberOffX
        fiberOffY = self.fiberOffY
        fixpos = self.fixpos
        fixedBH = self.fixedBH
        specialfit = self.specialfit
        interppm = self.interppm
        approx = self.approx

        phasemaps = self.phasemaps
        if phasemaps:
            northangle = self.northangle
            ddec = self.ddec
            dra = self.dra
        
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
        fluxRatioBG = theta[7]
        if fixedBG:
            alpha_bg = 3.
        else:
            alpha_bg = theta[8]
        phaseCenterRA = theta[9]
        phaseCenterDEC = theta[10]
        if use_opds:
            opd1 = theta[11]
            opd2 = theta[12]
            opd3 = theta[13]
            opd4 = theta[14]
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
        
        if phasemaps:
            cor_amp_sgr, cor_pha_sgr, cor_int_sgr = self.readPhasemaps(phaseCenterRA,
                                                                       phaseCenterDEC,
                                                                       fromFits=False, 
                                                                       northangle=northangle,
                                                                       dra=dra, ddec=ddec,
                                                                       interp=interppm)
            cor_amp_s2, cor_pha_s2, cor_int_s2 = self.readPhasemaps(dRA+phaseCenterRA, 
                                                                    dDEC+phaseCenterDEC,
                                                                    fromFits=False, 
                                                                    northangle=northangle,
                                                                    dra=dra, ddec=ddec,
                                                                    interp=interppm)

            pm_amp_sgr = np.array([[cor_amp_sgr[0], cor_amp_sgr[1]],
                                    [cor_amp_sgr[0], cor_amp_sgr[2]],
                                    [cor_amp_sgr[0], cor_amp_sgr[3]],
                                    [cor_amp_sgr[1], cor_amp_sgr[2]],
                                    [cor_amp_sgr[1], cor_amp_sgr[3]],
                                    [cor_amp_sgr[2], cor_amp_sgr[3]]])
            pm_pha_sgr = np.array([[cor_pha_sgr[0], cor_pha_sgr[1]],
                                    [cor_pha_sgr[0], cor_pha_sgr[2]],
                                    [cor_pha_sgr[0], cor_pha_sgr[3]],
                                    [cor_pha_sgr[1], cor_pha_sgr[2]],
                                    [cor_pha_sgr[1], cor_pha_sgr[3]],
                                    [cor_pha_sgr[2], cor_pha_sgr[3]]])
            pm_int_sgr = np.array([[cor_int_sgr[0], cor_int_sgr[1]],
                                    [cor_int_sgr[0], cor_int_sgr[2]],
                                    [cor_int_sgr[0], cor_int_sgr[3]],
                                    [cor_int_sgr[1], cor_int_sgr[2]],
                                    [cor_int_sgr[1], cor_int_sgr[3]],
                                    [cor_int_sgr[2], cor_int_sgr[3]]])
            pm_amp_s2 = np.array([[cor_amp_s2[0], cor_amp_s2[1]],
                                    [cor_amp_s2[0], cor_amp_s2[2]],
                                    [cor_amp_s2[0], cor_amp_s2[3]],
                                    [cor_amp_s2[1], cor_amp_s2[2]],
                                    [cor_amp_s2[1], cor_amp_s2[3]],
                                    [cor_amp_s2[2], cor_amp_s2[3]]])
            pm_pha_s2 = np.array([[cor_pha_s2[0], cor_pha_s2[1]],
                                    [cor_pha_s2[0], cor_pha_s2[2]],
                                    [cor_pha_s2[0], cor_pha_s2[3]],
                                    [cor_pha_s2[1], cor_pha_s2[2]],
                                    [cor_pha_s2[1], cor_pha_s2[3]],
                                    [cor_pha_s2[2], cor_pha_s2[3]]])   
            pm_int_s2 = np.array([[cor_int_s2[0], cor_int_s2[1]],
                                    [cor_int_s2[0], cor_int_s2[2]],
                                    [cor_int_s2[0], cor_int_s2[3]],
                                    [cor_int_s2[1], cor_int_s2[2]],
                                    [cor_int_s2[1], cor_int_s2[3]],
                                    [cor_int_s2[2], cor_int_s2[3]]])   
            
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
                
                opd_sgr = (pm_pha_sgr[i,0] - pm_pha_sgr[i,1])/360*wave
                opd_s2 = (pm_pha_s2[i,0] - pm_pha_s2[i,1])/360*wave
                s_SgrA -= opd_sgr
                s_S2 -= opd_s2
                
                cr1 = (pm_amp_s2[i,0] / pm_amp_sgr[i,0])**2
                cr2 = (pm_amp_s2[i,1] / pm_amp_sgr[i,1])**2
                
                cr_denom1 = (pm_int_s2[i,0] / pm_int_sgr[i,0])
                cr_denom2 = (pm_int_s2[i,0] / pm_int_sgr[i,0])
                
                if approx == "approx":
                    intSgrA = self.vis_intensity_approx(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                    intS2 = self.vis_intensity_approx(s_S2, alpha_S2, wave, dlambda[i,:])
                    intSgrA_center = self.vis_intensity_approx(0, alpha_SgrA, wave, dlambda[i,:])
                    intS2_center = self.vis_intensity_approx(0, alpha_S2, wave, dlambda[i,:])
                    intBG = self.vis_intensity_approx(0, alpha_bg, wave, dlambda[i,:])
                elif approx == "analytic":
                    intSgrA = self.vis_intensity(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                    intS2 = self.vis_intensity(s_S2, alpha_S2, wave, dlambda[i,:])
                    intSgrA_center = self.vis_intensity(0, alpha_SgrA, wave, dlambda[i,:])
                    intS2_center = self.vis_intensity(0, alpha_S2, wave, dlambda[i,:])
                    intBG = self.vis_intensity(0, alpha_bg, wave, dlambda[i,:])
                elif approx == "numeric":
                    intSgrA = self.vis_intensity_num(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                    intS2 = self.vis_intensity_num(s_S2, alpha_S2, wave, dlambda[i,:])
                    intSgrA_center = self.vis_intensity_num(0, alpha_SgrA, wave, dlambda[i,:])
                    intS2_center = self.vis_intensity_num(0, alpha_S2, wave, dlambda[i,:])
                    intBG = self.vis_intensity_num(0, alpha_bg, wave, dlambda[i,:])
                else:
                    raise ValueError('approx has to be approx, analytic or numeric')
                    
                vis[i,:] = ((intSgrA + 
                            np.sqrt(f_bl[i,0] * f_bl[i,1] * cr1 * cr2) * intS2)/
                            (np.sqrt(intSgrA_center + f_bl[i,0] * cr_denom1 * intS2_center 
                                    + fluxRatioBG * intBG) *
                             np.sqrt(intSgrA_center + f_bl[i,1] * cr_denom2 * intS2_center 
                                    + fluxRatioBG * intBG)))  

        else:
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
                
                if approx == "approx":
                    intSgrA = self.vis_intensity_approx(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                    intS2 = self.vis_intensity_approx(s_S2, alpha_S2, wave, dlambda[i,:])
                    intSgrA_center = self.vis_intensity_approx(0, alpha_SgrA, wave, dlambda[i,:])
                    intS2_center = self.vis_intensity_approx(0, alpha_S2, wave, dlambda[i,:])
                    intBG = self.vis_intensity_approx(0, alpha_bg, wave, dlambda[i,:])
                elif approx == "analytic":
                    intSgrA = self.vis_intensity(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                    intS2 = self.vis_intensity(s_S2, alpha_S2, wave, dlambda[i,:])
                    intSgrA_center = self.vis_intensity(0, alpha_SgrA, wave, dlambda[i,:])
                    intS2_center = self.vis_intensity(0, alpha_S2, wave, dlambda[i,:])
                    intBG = self.vis_intensity(0, alpha_bg, wave, dlambda[i,:])
                elif approx == "numeric":
                    intSgrA = self.vis_intensity_num(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                    intS2 = self.vis_intensity_num(s_S2, alpha_S2, wave, dlambda[i,:])
                    intSgrA_center = self.vis_intensity_num(0, alpha_SgrA, wave, dlambda[i,:])
                    intS2_center = self.vis_intensity_num(0, alpha_S2, wave, dlambda[i,:])
                    intBG = self.vis_intensity_num(0, alpha_bg, wave, dlambda[i,:])
                else:
                    raise ValueError('approx has to be approx, analytic or numeric')
                    
                vis[i,:] = ((intSgrA + 
                            np.sqrt(f_bl[i,0] * f_bl[i,1]) * intS2)/
                            (np.sqrt(intSgrA_center + f_bl[i,0] * intS2_center 
                                    + fluxRatioBG * intBG) *
                            np.sqrt(intSgrA_center + f_bl[i,1] * intS2_center 
                                    + fluxRatioBG * intBG)))

        visamp = np.abs(vis)
        visphi = np.angle(vis, deg=True)
        closure = np.zeros((4, len(wave)))
        closamp = np.zeros((4, len(wave)))
        for idx in range(4):
            closure[idx] = visphi[self.bispec_ind[idx,0]] + visphi[self.bispec_ind[idx,1]] - visphi[self.bispec_ind[idx,2]]
            closamp[idx] = visamp[self.bispec_ind[idx,0]] * visamp[self.bispec_ind[idx,1]] * visamp[self.bispec_ind[idx,2]]

        visphi = visphi + 360.*(visphi<-180.) - 360.*(visphi>180.)
        closure = closure + 360.*(closure<-180.) - 360.*(closure>180.)
        return visamp, visphi, closure, closamp
    
    
    def lnprob(self, theta, fitdata, u, v, wave, dlambda, lower, upper):
        if np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        return self.lnlike(theta, fitdata, u, v, wave, dlambda)
    
        
    def lnlike(self, theta, fitdata, u, v, wave, dlambda):       
        """
        Calculate the likelihood estimation for the MCMC run
        """
        # Model
        model_visamp, model_visphi, model_closure, model_closamp = self.calc_vis(theta,u,v,wave,dlambda)
        model_vis2 = model_visamp**2.
        
        #Data
        (visamp, visamp_error, visamp_flag,
         vis2, vis2_error, vis2_flag,
         closure, closure_error, closure_flag,
         visphi, visphi_error, visphi_flag,
         closamp, closamp_error, closamp_flag) = fitdata
        
        res_visamp = np.sum(-(model_visamp-visamp)**2/visamp_error**2*(1-visamp_flag))
        res_vis2 = np.sum(-(model_vis2-vis2)**2./vis2_error**2.*(1-vis2_flag))
        res_closamp = np.sum(-(model_closamp-closamp)**2/closamp_error**2*(1-closamp_flag))
        
        res_closure_1 = np.abs(model_closure-closure)
        res_closure_2 = 360-np.abs(model_closure-closure)
        check = np.abs(res_closure_1) < np.abs(res_closure_2)
        res_closure = res_closure_1*check + res_closure_2*(1-check)
        res_clos = np.sum(-res_closure**2./closure_error**2.*(1-closure_flag))
 
        res_visphi_1 = np.abs(model_visphi-visphi)
        res_visphi_2 = 360-np.abs(model_visphi-visphi)
        check = np.abs(res_visphi_1) < np.abs(res_visphi_2) 
        res_visphi = res_visphi_1*check + res_visphi_2*(1-check)
        res_phi = np.sum(-res_visphi**2./visphi_error**2.*(1-visphi_flag))

        ln_prob_res = 0.5 * (res_visamp * self.fit_for[0] + 
                             res_vis2 * self.fit_for[1] + 
                             res_clos * self.fit_for[2] + 
                             res_phi * self.fit_for[3] + 
                             res_closamp * self.fit_for[4])
        
        return ln_prob_res 
    
    
    
    def fitBinary(self, 
                  nthreads=4, 
                  nwalkers=500, 
                  nruns=500, 
                  bestchi=True,
                  bequiet=False,
                  fit_for=np.array([0.5,0.5,1.0,0.0,0.0]), 
                  nophases=False, 
                  approx='approx', 
                  donotfit=False, 
                  donotfittheta=None, 
                  onlypol1=False, 
                  initial=None, 
                  dRA=0., 
                  dDEC=0., 
                  dphRA=0.1, 
                  dphDec=0.1,
                  flagtill=3, 
                  flagfrom=13,
                  use_opds=False, 
                  fixedBG=True, 
                  noS2=True, 
                  fixpos=False, 
                  fixedBH=False, 
                  constant_f=True,
                  specialpar=np.array([0,0,0,0,0,0]), 
                  plot=True, 
                  plotres=True, 
                  createpdf=True, 
                  writeresults=True, 
                  redchi2=False,
                  phasemaps=False,
                  interppm=True, 
                  #smoothpm=True,
                  #smoothfwhm=65, 
                  simulate_pm=False):
        '''
        Binary fit to GRAVITY data
        Parameter:
        nthreads:       number of cores [4] 
        nwalkers:       number of walkers [500] 
        nruns:          number of MCMC runs [500] 
        bestchi:        Gives best chi2 (for True) or mcmc res as output [True]
        bequiet:        Suppresses ALL outputs
        fit_for:        weight of VA, V2, T3, VP, T3AMP [[0.5,0.5,1.0,0.0,0.0]] 
        nophases:       Does not fit phases, but still considers them for chi2
                        (for testing) [False]
        approx:         Kind of integration for visibilities (approx, numeric, analytic)
        donotfit:       Only gives fitting results for parameters from donotfittheta [False]
        donotfittheta:  has to be given for donotfit [None]
        onlypol1:       Only fits polarization 1 for split mode [False]
        initial:        Initial guess for fit [None]
        dRA:            Initial guess for dRA (taken from SOFFX if 0) [0]
        dDEC:           Initial guess for dDEC (taken from SOFFY if 0) [0]
        dphRA:          Initial guess for phase center RA [0]
        dphDec:         Initial guess for phase center DEC [0]
        flagtill:       Flag blue channels, has to be changed for not LOW [3] 
        flagfrom:       Flag red channels, has to be changed for not LOW [13]
        use_opds:       Fit OPDs [False] 
        fixedBG:        Fit for background power law [False]
        noS2:           Does not do anything if OFFX and OFFY=0
        fixpos:         Does nto fit the distance between the sources [False]
        fixedBH:        Fit for black hole power law [False]
        constant_f:     Constant coupling [True]
        specialpar:     Allows OPD for individual baseline [0,0,0,0,0,0]
        plot:           plot MCMC results [True]
        plotres:        plot fit result [True]
        createpdf:      Creates a pdf with fit results and all plots [True] 
        writeresults:  Write fit results in file [True]
        redchi2:        Gives redchi2 instead of chi2 [False]
        phasemaps:      Use Phasemaps for fit [False]
        interppm:       Interpolate Phasemaps [True]
        smoothpm:       Smooth the phasemaps by smoothfwhm [True]
        smoothfwhm:     FWHM for smoothing if smoothpm in mas [65] 
        simulate_pm:    Phasemaps for suimulated data, sets ACQ parameter to 0 [False]
        
        For a fit to two components: A (SgrA*) and B (S2)
        The possible fit properties are:
        dRA         Separation from A to B in RA [mas]
        dDEC        Separation from A to B in Dec [mas]
        f1          Flux ratio log(B/A) for telescope 1 (or all telescopes)
        f2          Flux ratio log(B/A) for telescope 2
        f3          Flux ratio log(B/A) for telescope 3
        f4          Flux ratio log(B/A) for telescope 4
        alpha A     spectral index of component A:  vSv = v^alpha 
        f BG        Flux ratio of background: BG/A
        alpha BG    spectral index of component BG:  vSv = v^alpha 
        PC RA       Offset of the phasecenter from field center in RA [mas]
        PC DEC      Offset of the phasecenter from field center in Dec [mas]
        OPD1        OPD for telescope 1 [mum]
        OPD2        OPD for telescope 2 [mum]
        OPD3        OPD for telescope 3 [mum]
        OPD4        OPD for telescope 4 [mum]
        special     OPD for individual baselines
        '''
        if self.resolution != 'LOW' and flagtill == 3 and flagfrom == 13:
            raise ValueError('Initial values for flagtill and flagfrom have to be changed if not low resolution')
        self.fit_for = fit_for
        self.constant_f = constant_f
        self.use_opds = use_opds
        self.fixedBG = fixedBG
        self.fixpos = fixpos
        self.fixedBH = fixedBH
        self.interppm = interppm
        self.approx = approx
        #self.smoothfwhm = smoothfwhm
        self.simulate_pm = simulate_pm
        self.bequiet = bequiet
        rad2as = 180 / np.pi * 3600
        if np.any(specialpar):
            self.specialfit = True
            self.specialfit_bl = specialpar
            if not bequiet:
                print('Specialfit parameter applied to BLs:')
                nonzero = np.nonzero(self.specialfit_bl)[0]
                print(*list(nonzero*self.specialfit_bl[nonzero]))
                print('\n')
        else:
            self.specialfit = False
        specialfit = self.specialfit
        
        self.phasemaps = phasemaps
        if phasemaps:
            self.loadPhasemaps(interp=interppm)
            
            header = fits.open(self.name)[0].header
            northangle1 = header['ESO QC ACQ FIELD1 NORTH_ANGLE']/180*math.pi
            northangle2 = header['ESO QC ACQ FIELD2 NORTH_ANGLE']/180*math.pi
            northangle3 = header['ESO QC ACQ FIELD3 NORTH_ANGLE']/180*math.pi
            northangle4 = header['ESO QC ACQ FIELD4 NORTH_ANGLE']/180*math.pi
            self.northangle = [northangle1, northangle2, northangle3, northangle4]

            ddec1 = header['ESO QC MET SOBJ DDEC1']
            ddec2 = header['ESO QC MET SOBJ DDEC2']
            ddec3 = header['ESO QC MET SOBJ DDEC3']
            ddec4 = header['ESO QC MET SOBJ DDEC4']
            self.ddec = [ddec1, ddec2, ddec3, ddec4]

            dra1 = header['ESO QC MET SOBJ DRA1']
            dra2 = header['ESO QC MET SOBJ DRA2']
            dra3 = header['ESO QC MET SOBJ DRA3']
            dra4 = header['ESO QC MET SOBJ DRA4']
            self.dra = [dra1, dra2, dra3, dra4]


        # Get data from file
        tel = fits.open(self.name)[0].header["TELESCOP"]
        if tel == 'ESO-VLTI-U1234':
            self.tel = 'UT'
        elif tel == 'ESO-VLTI-A1234':
            self.tel = 'AT'
        else:
            raise ValueError('Telescope not AT or UT, something wrong with input data')

        nwave = self.channel
        self.getIntdata(plot=False, flag=False)
        MJD = fits.open(self.name)[0].header["MJD-OBS"]
        u = self.u
        v = self.v
        wave = self.wlSC
            
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
            if not bequiet:
                print('Guess for RA & DEC from function as: %.2f, %.2f' % (dRA, dDEC))
        
        stname = self.name.find('GRAVI')
        if phasemaps:
            txtfilename = 'pm_binaryfit_' + self.name[stname:-5] + '.txt'
        else:
            txtfilename = 'binaryfit_' + self.name[stname:-5] + '.txt'
        if writeresults:
            txtfile = open(txtfilename, 'w')
            txtfile.write('# Results of binary fit for %s \n' % self.name[stname:])
            txtfile.write('# Lines are: Best chi2, MCMC result, MCMC error -, MCMC error + \n')
            txtfile.write('# Rowes are: dRA, dDEC, f1, f2, f3, f4, alpha flare, f BG, alpha BG, PC RA, PC DEC, OPD1, OPD2, OPD3, OPD4 \n')
            txtfile.write('# Parameter which are not fitted have 0.0 as error \n')
            txtfile.write('# MJD: %f \n' % MJD)
            txtfile.write('# OFFX: %f \n' % self.fiberOffX)
            txtfile.write('# OFFY: %f \n\n' % self.fiberOffY)

        self.wave = wave
        self.getDlambda()
        dlambda = self.dlambda
        results = []

        # Initial guesses
        if initial is not None:
            if len(initial) != 16:
                raise ValueError('Length of initial parameter list is not correct')
            size = 2
            dRA_init = np.array([initial[0],initial[0]-size,initial[0]+size])
            dDEC_init = np.array([initial[1],initial[1]-size,initial[1]+size])

            flux_ratio_1_init = np.array([np.log10(initial[2]), np.log10(0.01), np.log10(100.)])
            flux_ratio_2_init = np.array([np.log10(initial[3]), np.log10(0.001), np.log10(100.)])
            flux_ratio_3_init = np.array([np.log10(initial[4]), np.log10(0.001), np.log10(100.)])
            flux_ratio_4_init = np.array([np.log10(initial[5]), np.log10(0.001), np.log10(100.)])

            alpha_SgrA_init = np.array([initial[6],-5.,7.])
            flux_ratio_bg_init = np.array([initial[7],0.,20.])
            color_bg_init = np.array([initial[8],-5.,7.])

            size = 2
            phase_center_RA_init = np.array([initial[9],initial[9]-size,initial[9]+size])
            phase_center_DEC_init = np.array([initial[10],initial[10]-size,initial[10]+size])

            opd_max = 0.5 # maximum opd in microns (lambda/4)
            opd_1_init = [initial[11],-opd_max,opd_max]
            opd_2_init = [initial[12],-opd_max,opd_max]
            opd_3_init = [initial[13],-opd_max,opd_max]
            opd_4_init = [initial[14],-opd_max,opd_max]
            special_par = [initial[15], -2, 2]
        else:
            size = 4
            dRA_init = np.array([dRA,dRA-size,dRA+size])
            dDEC_init = np.array([dDEC,dDEC-size,dDEC+size])

            fr_start = np.log10(0.1)
            flux_ratio_1_init = np.array([fr_start, np.log10(0.01), np.log10(10.)])
            flux_ratio_2_init = np.array([fr_start, np.log10(0.001), np.log10(10.)])
            flux_ratio_3_init = np.array([fr_start, np.log10(0.001), np.log10(10.)])
            flux_ratio_4_init = np.array([fr_start, np.log10(0.001), np.log10(10.)])

            alpha_SgrA_init = np.array([-1.,-10.,10.])
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
                          flux_ratio_3_init[0],flux_ratio_4_init[0],alpha_SgrA_init[0],
                          flux_ratio_bg_init[0],color_bg_init[0],phase_center_RA_init[0],
                          phase_center_DEC_init[0],opd_1_init[0],opd_2_init[0],opd_3_init[0],opd_4_init[0],special_par[0]])

        # lower limit on fit parameters 
        theta_lower = np.array([dRA_init[1],dDEC_init[1],flux_ratio_1_init[1],flux_ratio_2_init[1],
                                flux_ratio_3_init[1],flux_ratio_4_init[1],alpha_SgrA_init[1],
                                flux_ratio_bg_init[1],color_bg_init[1],phase_center_RA_init[1],
                                phase_center_DEC_init[1],opd_1_init[1],opd_2_init[1],opd_3_init[1],opd_4_init[1], special_par[1]])

        # upper limit on fit parameters 
        theta_upper = np.array([dRA_init[2],dDEC_init[2],flux_ratio_1_init[2],flux_ratio_2_init[2],
                                flux_ratio_3_init[2],flux_ratio_4_init[2],alpha_SgrA_init[2],
                                flux_ratio_bg_init[2],color_bg_init[2],phase_center_RA_init[2],
                                phase_center_DEC_init[2],opd_1_init[2],opd_2_init[2],opd_3_init[2],opd_4_init[2], special_par[2]])

        theta_names = np.array(["dRA", "dDEC", "f1", "f2", "f3", "f4" , r"$\alpha_{flare}$", r"$f_{bg}$",
                                r"$\alpha_{bg}$", r"$RA_{PC}$", r"$DEC_{PC}$", "OPD1", "OPD2", "OPD3", "OPD4", "special"])
        theta_names_raw = np.array(["dRA", "dDEC", "f1", "f2", "f3", "f4" , "alpha flare", "f BG",
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
        if fixedBG:
            todel.append(8)
        if fit_for[3] == 0 or nophases:
            todel.append(9)
            todel.append(10)
        if not use_opds:
            todel.append(11)
            todel.append(12)
            todel.append(13)
            todel.append(14)
        if not specialfit:
            todel.append(15)
        ndof = ndim - len(todel)
        
        if donotfit:
            if donotfittheta is None:
                raise ValueError('If donotfit is True, fit values have to be given by donotfittheta')
            if len(donotfittheta) != ndim:
                print(theta_names_raw)
                raise ValueError('donotfittheta has to have %i parameters, see above' % ndim)
            if plot:
                raise ValueError('If donotfit is True, cannot create MCMC plots')
            if writeresults or createpdf:
                raise ValueError('If donotfit is True, writeresults and createpdf should be False')
            print('Will not fit the data, just print out the results for the given theta')
            #donotfittheta[2:6] = np.log10(donotfittheta[2:6])
            
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

            closamp_P = [self.t3ampSC_P1, self.t3ampSC_P2]
            closamp_error_P = [self.t3amperrSC_P1, self.t3amperrSC_P2]
            closamp_flag_P = [self.t3ampflagSC_P1, self.t3ampflagSC_P2]

            ndit = np.shape(self.visampSC_P1)[0]//6
            if not bequiet:
                print('NDIT = %i' % ndit)
            polnom = 2
        elif self.polmode == 'COMBINED':
            visamp_P = [self.visampSC]
            visamp_error_P = [self.visamperrSC]
            visamp_flag_P = [self.visampflagSC]
            
            vis2_P = [self.vis2SC]
            vis2_error_P = [self.vis2errSC]
            vis2_flag_P = [self.vis2flagSC]

            closure_P = [self.t3SC]
            closure_error_P = [self.t3errSC]
            closure_flag_P = [self.t3flagSC]
            
            visphi_P = [self.visphiSC]
            visphi_error_P = [self.visphierrSC]
            visphi_flag_P = [self.visampflagSC]

            closamp_P = [self.t3ampSC]
            closamp_error_P = [self.t3amperrSC]
            closamp_flag_P = [self.t3ampflagSC]

            ndit = np.shape(self.visampSC)[0]//6
            if not bequiet:
                print('NDIT = %i' % ndit)
            polnom = 1

        for dit in range(ndit):
            if writeresults and ndit > 1:
                txtfile.write('# DIT %i \n' % dit)
            if createpdf:
                savetime = str(datetime.now()).replace('-', '')
                savetime = savetime.replace(' ', '-')
                savetime = savetime.replace(':', '')
                self.savetime = savetime
                if phasemaps:
                    if ndit == 1:
                        pdffilename = 'pm_binaryfit_' + self.name[stname:-5] + '.pdf'
                    else:
                        pdffilename = 'pm_binaryfit_' + self.name[stname:-5] + '_DIT' + str(dit) + '.pdf'
                else:
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
                pdf.cell(40, 6, txt="Phasemaps", ln=0, align="L", border=0)
                pdf.cell(40, 6, txt=str(phasemaps), ln=0, align="L", border=0)
                pdf.cell(40, 6, txt="Integral solved by", ln=0, align="L", border=0)
                pdf.cell(40, 6, txt=approx, ln=1, align="L", border=0)
                #pdf.cell(40, 6, txt="Phasemaps smoothed", ln=0, align="L", border=0)
                #pdf.cell(40, 6, txt=str(smoothpm), ln=0, align="L", border=0)
                #pdf.cell(40, 6, txt="Smoothing FWHM", ln=0, align="L", border=0)
                #pdf.cell(40, 6, txt=str(smoothfwhm), ln=1, align="L", border=0)
                pdf.ln()
            
            if not bequiet and not donotfit:
                print('Run MCMC for DIT %i' % (dit+1))
            ditstart = dit*6
            ditstop = ditstart + 6
            t3ditstart = dit*4
            t3ditstop = t3ditstart + 4
            
            if onlypol1:
                polnom = 1
            for idx in range(polnom):
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
                closamp = closamp_P[idx][t3ditstart:t3ditstop]
                closamp_error = closamp_error_P[idx][t3ditstart:t3ditstop]
                closamp_flag = closamp_flag_P[idx][t3ditstart:t3ditstop]
                
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
                    closamp_flag[:,0:p] = True

                    visamp_flag[:,t:] = True
                    vis2_flag[:,t:] = True
                    visphi_flag[:,t:] = True
                    closure_flag[:,t:] = True
                    closamp_flag[:,t:] = True
                                        
                width = 1e-1
                pos = np.ones((nwalkers,ndim))
                for par in range(ndim):
                    if par in todel:
                        pos[:,par] = theta[par]
                    else:
                        pos[:,par] = theta[par] + width*np.random.randn(nwalkers)
                if not bequiet:
                    if not donotfit:
                        print('Run MCMC for Pol %i' % (idx+1))
                    else:
                        print('Pol %i' % (idx+1))
                fitdata = [visamp, visamp_error, visamp_flag,
                            vis2, vis2_error, vis2_flag,
                            closure, closure_error, closure_flag,
                            visphi, visphi_error, visphi_flag,
                            closamp, closamp_error, closamp_flag]
                
                self.fitstuff = [fitdata, u, v, wave, dlambda, theta]

                if not donotfit:
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
                        print("Mean acceptance fraction: %.2f"  
                              % np.mean(sampler.acceptance_fraction))
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
                else:
                    theta_result = donotfittheta
                    
                results.append(theta_result)
                fit_visamp, fit_visphi, fit_closure, fit_closamp = self.calc_vis(theta_result, u, v, wave, dlambda)
                fit_vis2 = fit_visamp**2.
                        
                res_visamp = fit_visamp-visamp
                res_vis2 = fit_vis2-vis2
                res_closamp = fit_closamp-closamp
                res_closure_1 = np.abs(fit_closure-closure)
                res_closure_2 = 360-np.abs(fit_closure-closure)
                check = np.abs(res_closure_1) < np.abs(res_closure_2) 
                res_closure = res_closure_1*check + res_closure_2*(1-check)
                res_visphi_1 = np.abs(fit_visphi-visphi)
                res_visphi_2 = 360-np.abs(fit_visphi-visphi)
                check = np.abs(res_visphi_1) < np.abs(res_visphi_2) 
                res_visphi = res_visphi_1*check + res_visphi_2*(1-check)

                redchi_visamp = np.sum(res_visamp**2./visamp_error**2.*(1-visamp_flag))
                redchi_vis2 = np.sum(res_vis2**2./vis2_error**2.*(1-vis2_flag))
                redchi_closure = np.sum(res_closure**2./closure_error**2.*(1-closure_flag))
                redchi_closamp = np.sum(res_closamp**2./closamp_error**2.*(1-closamp_flag))
                redchi_visphi = np.sum(res_visphi**2./visphi_error**2.*(1-visphi_flag))
                
                
                if redchi2:
                    redchi_visamp /= (visamp.size-np.sum(visamp_flag)-ndof)
                    redchi_vis2 /= (vis2.size-np.sum(vis2_flag)-ndof)
                    redchi_closure /= (closure.size-np.sum(closure_flag)-ndof)
                    redchi_closamp /= (closamp.size-np.sum(closamp_flag)-ndof)
                    redchi_visphi /= (visphi.size-np.sum(visphi_flag)-ndof)
                    chi2string = 'red. chi2'
                else:
                    chi2string = 'chi2'
                
                redchi = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi, redchi_closamp]
                if idx == 0:
                    redchi0 = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi, redchi_closamp]
                elif idx == 1:
                    redchi1 = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi, redchi_closamp]
                    
                if not bequiet:
                    print('\n')
                    print('ndof: %i' % (vis2.size-np.sum(vis2_flag)-ndof))
                    print(chi2string + " for visamp: %.2f" % redchi_visamp)
                    print(chi2string + " for vis2: %.2f" % redchi_vis2)
                    print(chi2string + " for visphi: %.2f" % redchi_visphi)
                    print(chi2string + " for closure: %.2f" % redchi_closure)
                    print(chi2string + " for closamp: %.2f" % redchi_closamp)
                    print('\n')
                    #print("average visamp error: %.2f" % 
                        #np.mean(visamp_error*(1-visamp_flag)))
                    #print("average vis2 error: %.2f" % 
                        #np.mean(vis2_error*(1-vis2_flag)))
                    #print("average closure error (deg): %.2f" % 
                        #np.mean(closure_error*(1-closure_flag)))
                    #print("average visphi error (deg): %.2f" % 
                        #np.mean(visphi_error*(1-visphi_flag)))
                
                if not donotfit:
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
                    self.plotFit(theta_result, fitdata, idx, createpdf=createpdf)
                if writeresults:
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
                            txtfile.write(', 0, 0, 0, 0, 0 \n')

                    for tdx, t in enumerate(percentiles[:,2]):
                        if tdx in todel:
                            txtfile.write(str(t*0.0))
                        else:
                            txtfile.write(str(t))
                        if tdx != (len(percentiles[:,1])-1):
                            txtfile.write(', ')
                        else:
                            txtfile.write(', 0, 0, 0, 0, 0\n')
                            
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
                    
                    if polnom == 2:
                        pdf.add_page()
                        pdf.cell(0, 10, txt="Polarization  2", ln=1, align="C", border='B')
                        pdf.ln()
                        pdf.image(pdfimages1[0], h=he, w=wi)
                        pdf.image(pdfimages1[1], h=115)
                    pdfcout = 2
                    
                if plotres:
                    titles = ['Vis Amp', 'Vis 2', 'Closure Phase', 'Visibility Phase', 'Closure Amp']
                    for pa in range(5):
                        if pa == 3 and not self.fit_for[pa]:
                            continue
                        if pa == 4 and not self.fit_for[pa]:
                            continue
                        pdf.add_page()
                        if polnom == 2:
                            text = '%s, %s: %.2f (P1), %.2f (P2)' % (titles[pa], chi2string, redchi0[pa], redchi1[pa])
                        else:
                            text = '%s, %s: %.2f' % (titles[pa], chi2string, redchi0[pa])
                        pdf.cell(0, 10, txt=text, ln=1, align="C", border='B')
                        pdf.ln()
                        pdf.image(pdfimages0[pdfcout+pa], w=150)
                        if polnom == 2:
                            pdf.image(pdfimages1[pdfcout+pa], w=150)
                
                if not bequiet:
                    print('Save pdf as %s' % pdffilename)
                pdf.output(pdffilename)
                files = glob.glob(savetime + '_pol?_?.png')
                for file in files:
                    os.remove(file)
        if writeresults:
            txtfile.close()
        if not bequiet:
            fitted = 1-(np.array(self.fit_for)==0)
            redchi0_f = np.sum(redchi0*fitted)
            if polnom < 2:
                redchi1 = np.zeros_like(redchi0)
            redchi1_f = np.sum(redchi1*fitted)
            redchi_f = redchi0_f + redchi1_f
            print('Combined %s of fitted data: %.3f' % (chi2string, redchi_f))
        if onlypol1 and ndit == 1:
            return theta_result
        else:
            return results
        
        
        
        
        
        
    ##################################################################################
    ##################################################################################
    ## Triple Fit
    ##################################################################################
    ##################################################################################
        
    def lnprob3(self, theta, fitdata, u, v, wave, dlambda, lower, upper):
        if np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        return self.lnlike3(theta, fitdata, u, v, wave, dlambda)
    
    
    def lnlike3(self, theta, fitdata, u, v, wave, dlambda):       
        """
        Calculate the likelihood estimation for the MCMC run
        """
        # Model
        model_visamp, model_visphi, model_closure, model_closamp = self.calc_vis3(theta,u,v,wave,dlambda)
        model_vis2 = model_visamp**2.
        
        #Data
        (visamp, visamp_error, visamp_flag,
         vis2, vis2_error, vis2_flag,
         closure, closure_error, closure_flag,
         visphi, visphi_error, visphi_flag,
         closamp, closamp_error, closamp_flag) = fitdata
        
        res_visamp = np.sum(-(model_visamp-visamp)**2/visamp_error**2*(1-visamp_flag))
        res_vis2 = np.sum(-(model_vis2-vis2)**2./vis2_error**2.*(1-vis2_flag))
        res_closamp = np.sum(-(model_closamp-closamp)**2/closamp_error**2*(1-closamp_flag))
        
        res_closure_1 = np.abs(model_closure-closure)
        res_closure_2 = 360-np.abs(model_closure-closure)
        check = np.abs(res_closure_1) < np.abs(res_closure_2)
        res_closure = res_closure_1*check + res_closure_2*(1-check)
        res_clos = np.sum(-res_closure**2./closure_error**2.*(1-closure_flag))
 
        res_visphi_1 = np.abs(model_visphi-visphi)
        res_visphi_2 = 360-np.abs(model_visphi-visphi)
        check = np.abs(res_visphi_1) < np.abs(res_visphi_2) 
        res_visphi = res_visphi_1*check + res_visphi_2*(1-check)
        res_phi = np.sum(-res_visphi**2./visphi_error**2.*(1-visphi_flag))

        ln_prob_res = 0.5 * (res_visamp * self.fit_for[0] + 
                             res_vis2 * self.fit_for[1] + 
                             res_clos * self.fit_for[2] + 
                             res_phi * self.fit_for[3] + 
                             res_closamp * self.fit_for[4])
        
        return ln_prob_res 
    
    
    
    def calc_vis3(self, theta, u, v, wave, dlambda):
        mas2rad = 1e-3 / 3600 / 180 * np.pi
        rad2mas = 180 / np.pi * 3600 * 1e3
        
        fixedBG = self.fixedBG
        fiberOffX = self.fiberOffX
        fiberOffY = self.fiberOffY
        fixpos = self.fixpos
        fixedBH = self.fixedBH
        approx = self.approx

        if fixpos:
            dRA = self.fiberOffX
            dDEC = self.fiberOffY
        else:
            dRA = theta[0]
            dDEC = theta[1]
        fluxRatio = theta[2]
        
        dRA2 = theta[3]
        dDEC2 = theta[4]
        fluxRatio2 = theta[5]
        
        if fixedBH:
            alpha_SgrA = -0.5
        else:
            alpha_SgrA = theta[6]

        fluxRatioBG = theta[7]
        if fixedBG:
            alpha_bg = 3.
        else:
            alpha_bg = theta[8]
        
        if self.fit_for[3] == 0:
            pc_RA = 0
            pc_DEC = 0
        else:
            pc_RA = theta[9]
            pc_DEC = theta[10]
        alpha_S = 3
        
        f = 10**fluxRatio
        f2 = 10**fluxRatio2
        
        vis = np.zeros((6,len(wave))) + 0j
        for i in range(0,6):
            s_SgrA = ((pc_RA)*u[i] + (pc_DEC)*v[i]) * mas2rad * 1e6
            s_S1 = ((dRA+pc_RA)*u[i] + (dDEC+pc_DEC)*v[i]) * mas2rad * 1e6
            s_S2 = ((dRA2+pc_RA)*u[i] + (dDEC2+pc_DEC)*v[i]) * mas2rad * 1e6
            
            if approx == "approx":
                intSgrA = self.vis_intensity_approx(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                intSgrA_center = self.vis_intensity_approx(0, alpha_SgrA, wave, dlambda[i,:])

                intS1 = self.vis_intensity_approx(s_S1, alpha_S, wave, dlambda[i,:])
                intS1_center = self.vis_intensity_approx(0, alpha_S, wave, dlambda[i,:])

                intS2 = self.vis_intensity_approx(s_S2, alpha_S, wave, dlambda[i,:])
                intS2_center = self.vis_intensity_approx(0, alpha_S, wave, dlambda[i,:])

                intBG = self.vis_intensity_approx(0, alpha_bg, wave, dlambda[i,:])
                
            elif approx == "analytic":
                intSgrA = self.vis_intensity(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                intSgrA_center = self.vis_intensity(0, alpha_SgrA, wave, dlambda[i,:])

                intS1 = self.vis_intensity(s_S1, alpha_S, wave, dlambda[i,:])
                intS1_center = self.vis_intensity(0, alpha_S, wave, dlambda[i,:])

                intS2 = self.vis_intensity(s_S2, alpha_S, wave, dlambda[i,:])
                intS2_center = self.vis_intensity(0, alpha_S, wave, dlambda[i,:])

                intBG = self.vis_intensity(0, alpha_bg, wave, dlambda[i,:])                
                
            elif approx == "numeric":
                intSgrA = self.vis_intensity_num(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                intSgrA_center = self.vis_intensity_num(0, alpha_SgrA, wave, dlambda[i,:])

                intS1 = self.vis_intensity_num(s_S1, alpha_S, wave, dlambda[i,:])
                intS1_center = self.vis_intensity_num(0, alpha_S, wave, dlambda[i,:])

                intS2 = self.vis_intensity_num(s_S2, alpha_S, wave, dlambda[i,:])
                intS2_center = self.vis_intensity_num(0, alpha_S, wave, dlambda[i,:])

                intBG = self.vis_intensity_num(0, alpha_bg, wave, dlambda[i,:])                
                intBG = self.vis_intensity_num(0, alpha_bg, wave, dlambda[i,:])

            else:
                raise ValueError('approx has to be approx, analytic or numeric')
            
            vis[i,:] = ((intSgrA + f*intS1 + f2*intS2)/
                        (intSgrA_center + f*intS1_center + f2*intS2_center + fluxRatioBG*intBG))

        visamp = np.abs(vis)
        visphi = np.angle(vis, deg=True)
        closure = np.zeros((4, len(wave)))
        closamp = np.zeros((4, len(wave)))
        for idx in range(4):
            closure[idx] = visphi[self.bispec_ind[idx,0]] + visphi[self.bispec_ind[idx,1]] - visphi[self.bispec_ind[idx,2]]
            closamp[idx] = visamp[self.bispec_ind[idx,0]] * visamp[self.bispec_ind[idx,1]] * visamp[self.bispec_ind[idx,2]]

        visphi = visphi + 360.*(visphi<-180.) - 360.*(visphi>180.)
        closure = closure + 360.*(closure<-180.) - 360.*(closure>180.)
        return visamp, visphi, closure, closamp



    def fitTriple(self, 
                  dRA2, 
                  dDEC2, 
                  fr2,
                  nthreads=4, 
                  nwalkers=500, 
                  nruns=500,
                  bestchi=True, 
                  bequiet=False, 
                  fit_for=np.array([0.5,0.5,1.0,0.0,0.0]),
                  approx='approx',
                  donotfit=False, 
                  donotfittheta=None, 
                  onlypol1=False, 
                  initial=None,
                  dRA=0., 
                  dDEC=0., 
                  dphRA=0.1, 
                  dphDec=0.1,
                  flagtill=3, 
                  flagfrom=13, 
                  fixedBG=True, 
                  noS2=True, 
                  fixpos=False, 
                  fixedBH=False, 
                  plot=True, 
                  plotres=True, 
                  createpdf=True,
                  redchi2=False):
        '''
        Tripple fit to GRAVITY data, reduced version of binary fit
        Parameter:
        dRA2:           Initial guess for position of 3rd source
        dDEC2:          Initial guess for position of 3rd source 
        fr2:            Initial guess for flux ratio of 3rd source
        
        nthreads:       number of cores [4] 
        nwalkers:       number of walkers [500] 
        nruns:          number of MCMC runs [500] 
        bestchi:        Gives best chi2 (for True) or mcmc res as output [True]
        bequiet:        Suppresses ALL outputs
        fit_for:        weight of VA, V2, T3, VP, T3AMP [[0.5,0.5,1.0,0.0,0.0]] 
        approx:         Kind of integration for visibilities (approx, numeric, analytic)
        donotfit:       Only gives fitting results for parameters from donotfittheta [False]
        donotfittheta:  has to be given for donotfit [None]
        onlypol1:       Only fits polarization 1 for split mode [False]
        initial:        Initial guess for fit [None]
        dRA:            Initial guess for dRA (taken from SOFFX if 0) [0]
        dDEC:           Initial guess for dDEC (taken from SOFFY if 0) [0]
        dphRA:          Initial guess for phase center RA [0]
        dphDec:         Initial guess for phase center DEC [0]
        flagtill:       Flag blue channels, has to be changed for not LOW [3] 
        flagfrom:       Flag red channels, has to be changed for not LOW [13]
        fixedBG:        Fir for background power law [False]
        noS2:           Does not do anything if OFFX and OFFY=0
        fixpos:         Does not fit the distance between the sources [False]
        fixedBH:        Fit for black hole power law [False]
        plot:           plot MCMC results [True]
        plotres:        plot fit result [True]
        createpdf:      Creates a pdf with fit results and all plots [True] 
        redchi2:        Gives redchi2 instead of chi2 [False]
        '''
        if self.resolution != 'LOW' and flagtill == 3 and flagfrom == 13:
            raise ValueError('Initial values for flagtill and flagfrom have to be changed if not low resolution')
        self.fit_for = fit_for
        self.fixedBG = fixedBG
        self.fixpos = fixpos
        self.fixedBH = fixedBH
        self.approx = approx
        rad2as = 180 / np.pi * 3600
        
        nwave = self.channel
        self.getIntdata(plot=False, flag=False)
        MJD = fits.open(self.name)[0].header["MJD-OBS"]
        u = self.u
        v = self.v
        wave = self.wlSC
            
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
            
        self.wave = wave
        self.getDlambda()
        dlambda = self.dlambda
        results = []
        
        # Initial guesses
        if initial is not None:
            if len(initial) != 11:
                raise ValueError('Length of initial parameter list is not correct')
            size = 2
            dRA_init = np.array([initial[0],initial[0]-size,initial[0]+size])
            dDEC_init = np.array([initial[1],initial[1]-size,initial[1]+size])
            flux_ratio_init = np.array([np.log10(initial[2]), np.log10(0.01), np.log10(100.)])

            dRA2_init = np.array([initial[3],initial[0]-size,initial[0]+size])
            dDEC2_init = np.array([initial[4],initial[1]-size,initial[1]+size])
            flux_ratio2_init = np.array([np.log10(initial[5]), np.log10(0.01), np.log10(100.)])

            alpha_SgrA_init = np.array([initial[6],-5.,7.])
            flux_ratio_bg_init = np.array([initial[7],0.,20.])
            color_bg_init = np.array([initial[8],-5.,7.])

            size = 2
            phase_center_RA_init = np.array([initial[9],initial[9]-size,initial[9]+size])
            phase_center_DEC_init = np.array([initial[10],initial[10]-size,initial[10]+size])

        else:
            size = 4
            dRA_init = np.array([dRA,dRA-size,dRA+size])
            dDEC_init = np.array([dDEC,dDEC-size,dDEC+size])

            dRA2_init = np.array([dRA2,dRA2-size,dRA2+size])
            dDEC2_init = np.array([dDEC2,dDEC2-size,dDEC2+size])

            fr_start = np.log10(0.1)
            flux_ratio_init = np.array([fr_start, np.log10(0.01), np.log10(100.)])
            fr2_start = np.log10(fr2)
            flux_ratio2_init = np.array([fr2_start, np.log10(0.01), np.log10(100.)])

            alpha_SgrA_init = np.array([-1.,-10.,10.])
            flux_ratio_bg_init = np.array([0.1,0.,20.])
            color_bg_init = np.array([3.,-10.,10.])

            size = 5
            phase_center_RA_init = np.array([dphRA,dphRA-size,dphRA+size])
            phase_center_DEC_init = np.array([dphDec,dphDec-size,dphDec+size])
            
        # initial fit parameters 
        theta = np.array([dRA_init[0], dDEC_init[0], flux_ratio_init[0],
                          dRA2_init[0], dDEC2_init[0], flux_ratio2_init[0],
                          alpha_SgrA_init[0], flux_ratio_bg_init[0], color_bg_init[0], 
                          phase_center_RA_init[0], phase_center_DEC_init[0]])

        # lower limit on fit parameters 
        theta_lower = np.array([dRA_init[1], dDEC_init[1], flux_ratio_init[1],
                                dRA2_init[1], dDEC2_init[1], flux_ratio2_init[1],
                                alpha_SgrA_init[1], flux_ratio_bg_init[1], color_bg_init[1], 
                                phase_center_RA_init[1], phase_center_DEC_init[1]])

        # upper limit on fit parameters 
        theta_upper = np.array([dRA_init[2], dDEC_init[2], flux_ratio_init[2],
                                dRA2_init[2], dDEC2_init[2], flux_ratio2_init[2],
                                alpha_SgrA_init[2], flux_ratio_bg_init[2], color_bg_init[2], 
                                phase_center_RA_init[2], phase_center_DEC_init[2]])

        theta_names = np.array(["dRA", "dDEC", "fr", "dRA2", "dDEC2", "fr2",
                                r"$\alpha_{flare}$", r"$f_{bg}$", r"$\alpha_{bg}$", 
                                r"$RA_{PC}$", r"$DEC_{PC}$"])
        theta_names_raw = np.array(["dRA", "dDEC", "fr", "dRA2", "dDEC2", "fr2",
                                    "alpha flare", "f BG", "alpha BG", "PC RA", "PC DEC"])
        

        ndim = len(theta)
        todel = []
        if fixpos:
            todel.append(0)
            todel.append(1)
        if fixedBH:
            todel.append(6)
        if fixedBG:
            todel.append(8)
        if fit_for[3] == 0:
            todel.append(9)
            todel.append(10)
        ndof = ndim - len(todel)

        if donotfit:
            if donotfittheta is None:
                raise ValueError('If donotfit is True, fit values have to be given by donotfittheta')
            if len(donotfittheta) != ndim:
                print(theta_names_raw)
                raise ValueError('donotfittheta has to have %i parameters, see above' % ndim)
            if plot:
                raise ValueError('If donotfit is True, cannot create MCMC plots')
            if writeresults or createpdf:
                raise ValueError('If donotfit is True, writeresults and createpdf should be False')
            print('Will not fit the data, just print out the results for the given theta')
            
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

            closamp_P = [self.t3ampSC_P1, self.t3ampSC_P2]
            closamp_error_P = [self.t3amperrSC_P1, self.t3amperrSC_P2]
            closamp_flag_P = [self.t3ampflagSC_P1, self.t3ampflagSC_P2]

            ndit = np.shape(self.visampSC_P1)[0]//6
            if not bequiet:
                print('NDIT = %i' % ndit)
            polnom = 2
        elif self.polmode == 'COMBINED':
            visamp_P = [self.visampSC]
            visamp_error_P = [self.visamperrSC]
            visamp_flag_P = [self.visampflagSC]
            
            vis2_P = [self.vis2SC]
            vis2_error_P = [self.vis2errSC]
            vis2_flag_P = [self.vis2flagSC]

            closure_P = [self.t3SC]
            closure_error_P = [self.t3errSC]
            closure_flag_P = [self.t3flagSC]
            
            visphi_P = [self.visphiSC]
            visphi_error_P = [self.visphierrSC]
            visphi_flag_P = [self.visampflagSC]

            closamp_P = [self.t3ampSC]
            closamp_error_P = [self.t3amperrSC]
            closamp_flag_P = [self.t3ampflagSC]

            ndit = np.shape(self.visampSC)[0]//6
            if not bequiet:
                print('NDIT = %i' % ndit)
            polnom = 1

        for dit in range(ndit):
            if not bequiet and not donotfit:
                print('Run MCMC for DIT %i' % (dit+1))
            ditstart = dit*6
            ditstop = ditstart + 6
            t3ditstart = dit*4
            t3ditstop = t3ditstart + 4
            
            if onlypol1:
                polnom = 1
                
            for idx in range(polnom):
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
                closamp = closamp_P[idx][t3ditstart:t3ditstop]
                closamp_error = closamp_error_P[idx][t3ditstart:t3ditstop]
                closamp_flag = closamp_flag_P[idx][t3ditstart:t3ditstop]
                
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
                    closamp_flag[:,0:p] = True

                    visamp_flag[:,t:] = True
                    vis2_flag[:,t:] = True
                    visphi_flag[:,t:] = True
                    closure_flag[:,t:] = True
                    closamp_flag[:,t:] = True
                                        
                width = 1e-1
                pos = np.ones((nwalkers,ndim))
                for par in range(ndim):
                    if par in todel:
                        pos[:,par] = theta[par]
                    else:
                        pos[:,par] = theta[par] + width*np.random.randn(nwalkers)
                if not bequiet:
                    if not donotfit:
                        print('Run MCMC for Pol %i' % (idx+1))
                    else:
                        print('Pol %i' % (idx+1))
                fitdata = [visamp, visamp_error, visamp_flag,
                            vis2, vis2_error, vis2_flag,
                            closure, closure_error, closure_flag,
                            visphi, visphi_error, visphi_flag,
                            closamp, closamp_error, closamp_flag]
                
                self.fitstuff = [fitdata, u, v, wave, dlambda, theta]

                if not donotfit:
                    if nthreads == 1:
                        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob3, 
                                                        args=(fitdata, u, v, wave,
                                                                dlambda, theta_lower,
                                                                theta_upper))
                        if bequiet:
                            sampler.run_mcmc(pos, nruns, progress=False)
                        else:
                            sampler.run_mcmc(pos, nruns, progress=True)
                    else:
                        with Pool(processes=nthreads) as pool:
                            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob3, 
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
                        print("Mean acceptance fraction: %.2f"  %
                              np.mean(sampler.acceptance_fraction))
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
                        plt.show()                            

                    if nruns > 299:
                        fl_samples = samples[:, -200:, :].reshape((-1, ndim))
                        fl_clsamples = clsamples[:, -200:, :].reshape((-1, cldim))                
                    elif nruns > 199:
                        fl_samples = samples[:, -100:, :].reshape((-1, ndim))
                        fl_clsamples = clsamples[:, -100:, :].reshape((-1, cldim))   
                    else:
                        fl_samples = samples.reshape((-1, ndim))
                        fl_clsamples = clsamples.reshape((-1, cldim))

                    if plot:
                        ranges = np.percentile(fl_clsamples, [3, 97], axis=0).T
                        fig = corner.corner(fl_clsamples, quantiles=[0.16, 0.5, 0.84],
                                            truths=clmostprop, labels=cllabels)
                        plt.show()

                    # get the actual fit
                    theta_fit = np.percentile(fl_samples, [50], axis=0).T
                    if bestchi:
                        theta_result = mostprop
                    else:
                        theta_result = theta_fit
                else:
                    theta_result = donotfittheta

                results.append(theta_result)
                fit_visamp, fit_visphi, fit_closure, fit_closamp = self.calc_vis3(theta_result, u, v, wave, dlambda)
                fit_vis2 = fit_visamp**2.
                        
                res_visamp = fit_visamp-visamp
                res_vis2 = fit_vis2-vis2
                res_closamp = fit_closamp-closamp
                res_closure_1 = np.abs(fit_closure-closure)
                res_closure_2 = 360-np.abs(fit_closure-closure)
                check = np.abs(res_closure_1) < np.abs(res_closure_2) 
                res_closure = res_closure_1*check + res_closure_2*(1-check)
                res_visphi_1 = np.abs(fit_visphi-visphi)
                res_visphi_2 = 360-np.abs(fit_visphi-visphi)
                check = np.abs(res_visphi_1) < np.abs(res_visphi_2) 
                res_visphi = res_visphi_1*check + res_visphi_2*(1-check)

                redchi_visamp = np.sum(res_visamp**2./visamp_error**2.*(1-visamp_flag))
                redchi_vis2 = np.sum(res_vis2**2./vis2_error**2.*(1-vis2_flag))
                redchi_closure = np.sum(res_closure**2./closure_error**2.*(1-closure_flag))
                redchi_closamp = np.sum(res_closamp**2./closamp_error**2.*(1-closamp_flag))
                redchi_visphi = np.sum(res_visphi**2./visphi_error**2.*(1-visphi_flag))
                
                
                if redchi2:
                    redchi_visamp /= (visamp.size-np.sum(visamp_flag)-ndof)
                    redchi_vis2 /= (vis2.size-np.sum(vis2_flag)-ndof)
                    redchi_closure /= (closure.size-np.sum(closure_flag)-ndof)
                    redchi_closamp /= (closamp.size-np.sum(closamp_flag)-ndof)
                    redchi_visphi /= (visphi.size-np.sum(visphi_flag)-ndof)
                    chi2string = 'red. chi2'
                else:
                    chi2string = 'chi2'
                
                redchi = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi, redchi_closamp]
                if idx == 0:
                    redchi0 = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi, redchi_closamp]
                elif idx == 1:
                    redchi1 = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi, redchi_closamp]
                    
                if not bequiet:
                    print('\n')
                    print('ndof: %i' % (vis2.size-np.sum(vis2_flag)-ndof))
                    print(chi2string + " for visamp: %.2f" % redchi_visamp)
                    print(chi2string + " for vis2: %.2f" % redchi_vis2)
                    print(chi2string + " for visphi: %.2f" % redchi_visphi)
                    print(chi2string + " for closure: %.2f" % redchi_closure)
                    print(chi2string + " for closamp: %.2f" % redchi_closamp)
                    print('\n')

                if not donotfit:
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

                if plotres:
                    self.plotFit(theta_result, fitdata, idx, createpdf=False, mode='triple')

        if not bequiet:
            fitted = 1-(np.array(self.fit_for)==0)
            redchi0_f = np.sum(redchi0*fitted)
            if polnom < 2:
                redchi1 = np.zeros_like(redchi0)
            redchi1_f = np.sum(redchi1*fitted)
            redchi_f = redchi0_f + redchi1_f
            print('Combined %s of fitted data: %.3f' % (chi2string, redchi_f))
        if onlypol1 and ndit == 1:
            return theta_result
        else:
            return results
        
        
    def plotFit(self, theta, fitdata, idx=0, createpdf=False, mode='binary'):
        """
        Calculates the theoretical interferometric data for the given parameters in theta
        and plots them together with the data in fitdata.
        Mainly used in fitBinary as result plots.
        """
        rad2as = 180 / np.pi * 3600
        
        (visamp, visamp_error, visamp_flag, 
         vis2, vis2_error, vis2_flag, 
         closure, closure_error, closure_flag, 
         visphi, visphi_error, visphi_flag,
         closamp, closamp_error, closamp_flag) = fitdata
        wave = self.wlSC
        dlambda = self.dlambda
        if self.phasemaps:
            wave_model = wave
        else:
            wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
        dlambda_model = np.zeros((6,len(wave_model)))
        for i in range(0,6):
            dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
            
        # Fit
        u = self.u
        v = self.v
        magu = np.sqrt(u**2.+v**2.)
        if mode == 'binary':
            (model_visamp_full, model_visphi_full, 
            model_closure_full, model_closamp_full)  = self.calc_vis(theta, u, v, wave_model, dlambda_model)
        elif mode == 'triple':
            (model_visamp_full, model_visphi_full, 
            model_closure_full, model_closamp_full)  = self.calc_vis3(theta, u, v, wave_model, dlambda_model)            
        else:
            raise ValueError('Plot mode has to be binary or triple')
        model_vis2_full = model_visamp_full**2.
        
        magu_as = self.spFrequAS
        
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
                        color=self.colors_baseline[i], alpha=0.5, label=self.baseline_labels[i])
            plt.plot(magu_as_model[i,:], model_visamp_full[i,:],
                     color='k', zorder=100)
        plt.ylabel('visibility modulus')
        plt.ylim(-0.1,1.1)
        plt.xlabel('spatial frequency (1/arcsec)')
        plt.legend()
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
                        color=self.colors_baseline[i],alpha=0.5, label=self.baseline_labels[i])
            plt.plot(magu_as_model[i,:], model_vis2_full[i,:],
                     color='k', zorder=100)
        plt.xlabel('spatial frequency (1/arcsec)')
        plt.ylabel('visibility squared')
        plt.ylim(-0.1,1.1)
        plt.legend()
        if createpdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_6.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        # T3
        for i in range(0,4):
            max_u_as_model = self.max_spf[i]/(wave_model*1.e-6) / rad2as
            plt.errorbar(self.spFrequAS_T3[i,:], closure[i,:]*(1-closure_flag)[i],
                         closure_error[i,:]*(1-closure_flag)[i],
                         color=self.colors_closure[i],ls='', lw=1, alpha=0.5, capsize=0)
            plt.scatter(self.spFrequAS_T3[i,:], closure[i,:]*(1-closure_flag)[i],
                        color=self.colors_closure[i], alpha=0.5, label=self.closure_labels[i])
            plt.plot(max_u_as_model, model_closure_full[i,:], 
                     color='k', zorder=100)
        plt.xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
        plt.ylabel('closure phase (deg)')
        plt.legend()
        if createpdf:
            plt.title('Polarization %i' % (idx + 1))
            pdfname = '%s_pol%i_7.png' % (savetime, idx)
            plt.savefig(pdfname)
            plt.close()
        else:
            plt.show()
        
        # VisPhi
        if self.fit_for[3]:
            for i in range(0,6):
                plt.errorbar(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i], 
                            visphi_error[i,:]*(1-visphi_flag)[i],
                            color=self.colors_baseline[i], ls='', lw=1, alpha=0.5, capsize=0)
                plt.scatter(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i],
                            color=self.colors_baseline[i], alpha=0.5, label=self.baseline_labels[i])
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
            
        # T3amp
        if self.fit_for[4]:
            for i in range(0,4):
                max_u_as_model = self.max_spf[i]/(wave_model*1.e-6) / rad2as
                plt.errorbar(self.spFrequAS_T3[i,:], closamp[i,:]*(1-closamp_flag)[i],
                            closamp_error[i,:]*(1-closamp_flag)[i],
                            color=self.colors_closure[i],ls='', lw=1, alpha=0.5, capsize=0)
                plt.scatter(self.spFrequAS_T3[i,:], closamp[i,:]*(1-closamp_flag)[i],
                            color=self.colors_closure[i], alpha=0.5)
                plt.plot(max_u_as_model, model_closamp_full[i,:], 
                        color='k', zorder=100)
            plt.xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
            plt.ylabel('closure Amplitude')
            if createpdf:
                plt.title('Polarization %i' % (idx + 1))
                pdfname = '%s_pol%i_9.png' % (savetime, idx)
                plt.savefig(pdfname)
                plt.close()
            else:
                plt.show()

        
        
        
        
        
    ##################################################################################
    ##################################################################################
    ## Unary Fit
    ##################################################################################
    ################################################################################## 
    
    def simulateUnaryphases(self, dRa, dDec, specialfit=False, specialpar=None, plot=False, compare=True, uvind=False):
        """
        Test function to see how a given phasecenter shift would look like
        """
        self.fixedBG = True
        self.fixedBH = True
        self.noBG = True
        self.use_opds = False
        self.specialfit = specialfit
        self.specialfit_bl = np.array([1,1,0,0,-1,-1])
        self.michistyle = False
        
        rad2as = 180 / np.pi * 3600
        
        nwave = self.channel
        if nwave != 14:
            raise ValueError('Only usable for 14 channels')
        self.getIntdata(plot=False, flag=False)
        u = self.u
        v = self.v
        wave = self.wlSC_P1
        self.getDlambda()
        dlambda = self.dlambda
                
        theta = [dRa, dDec, 0, 0, 0, 0, 0, 0, 0, specialpar]
        fit_visphi = self.calc_vis_unary(theta, u, v, wave, dlambda)
        
        if compare:
            visphi = self.visphiSC_P1[:6]
            visphi_error = self.visphierrSC_P1[:6]
            visphi_flag = self.visampflagSC_P1[:6]
            visphi_flag[:,0:2] = True
            visphi_flag[:,12:] = True
            fitdata = [visphi, visphi_error, visphi_flag]
            self.plotFitUnary(theta, fitdata, u, v, 1,
                              createpdf=False, uvind=uvind)
            
            
            
            res_visphi_1 = fit_visphi-visphi
            #res_visphi_2 = 360-(fit_visphi-visphi)
            #check = np.abs(res_visphi_1) < np.abs(res_visphi_2) 
            #res_visphi = res_visphi_1*check + res_visphi_2*(1-check)
            chi2 = np.sum(res_visphi_1**2./visphi_error**2.*(1-visphi_flag))
            print(chi2)
            return visphi
            
        else:
            if plot:
                wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
                dlambda_model = np.zeros((6,len(wave_model)))
                for i in range(0,6):
                    dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
                model_visphi_full  = self.calc_vis_unary(theta, u, v, wave_model, dlambda_model)
                
                #magu = np.sqrt(u**2.+v**2.)
                #u_as = np.zeros((len(u),len(wave)))
                #v_as = np.zeros((len(v),len(wave)))
                #for i in range(0,len(u)):
                    #u_as[i,:] = u[i]/(wave*1.e-6) / rad2as
                    #v_as[i,:] = v[i]/(wave*1.e-6) / rad2as
                #magu_as = np.sqrt(u_as**2.+v_as**2.)
                magu_as = self.spFrequAS

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
        michistyle = self.michistyle
        approx = self.approx
        
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
                
        for i in range(0,6):
            # pc in mas -> mas2rad -> pc in rad
            # uv in m -> *1e6 -> uv in mum
            # s in mum
            s_SgrA = ((phaseCenterRA)*u[i] + (phaseCenterDEC)*v[i]) * mas2rad * 1e6
            
            if use_opds:
                s_SgrA = s_SgrA + opd_bl[i,0] - opd_bl[i,1]
            if specialfit:
                s_SgrA = s_SgrA + sp_bl[i]

            if michistyle:
                s_sgra_ul = s_SgrA / wave
                vis[i,:] = np.exp(-2j*np.pi*s_sgra_ul)
            
            else:
                # interferometric intensities of all components
                if approx == "approx":
                    intSgrA = self.vis_intensity_approx(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                    intSgrA_center = self.vis_intensity_approx(0, alpha_SgrA, wave, dlambda[i,:])
                    intBG = self.vis_intensity_approx(0, alpha_bg, wave, dlambda[i,:])
                elif approx == "analytic":
                    intSgrA = self.vis_intensity(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                    intSgrA_center = self.vis_intensity(0, alpha_SgrA, wave, dlambda[i,:])
                    intBG = self.vis_intensity(0, alpha_bg, wave, dlambda[i,:])
                elif approx == "numeric":
                    intSgrA = self.vis_intensity_num(s_SgrA, alpha_SgrA, wave, dlambda[i,:])
                    intSgrA_center = self.vis_intensity_num(0, alpha_SgrA, wave, dlambda[i,:])
                    intBG = self.vis_intensity_num(0, alpha_bg, wave, dlambda[i,:])
                else:
                    raise ValueError('approx has to be approx, analytic or numeric')                
                
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
        
        self.getDlambda()
        dlambda = self.dlambda

        visphi = self.calc_vis_unary(theta, u, v, wave, dlambda)
        return visphi
        

    def fitUnary(self, 
                 nthreads=4, 
                 nwalkers=500, 
                 nruns=500, 
                 bestchi=True,
                 bequiet=False, 
                 michistyle=False,
                 approx='approx', 
                 onlypol1=False, 
                 mindatapoints=3,
                 flagtill=2, 
                 flagfrom=12, 
                 dontfit=None, 
                 dontfitbl=None, 
                 noS2=False, 
                 fixedBG=True, 
                 fixedBH=True, 
                 noBG=True,
                 fitopds=np.array([0,0,0,0]), 
                 specialpar=np.array([0,0,0,0,0,0]), 
                 plot=True, 
                 plotres=True, 
                 writeresults=True,
                 createpdf=True, 
                 writefitdiff=False,
                 returnPos=False,
                 phaseinput=None):
        """
        Does a MCMC unary fit on the phases of the data.
        Parameter:
        nthreads:       number of cores [4] 
        nwalkers:       number of walkers [500] 
        nruns:          number of MCMC runs [500] 
        bestchi:        Gives best chi2 (for True) or mcmc res as output [True]
        bequiet:        Suppresses ALL outputs [False]
        michistyle:     Uses a simple exponential fit instead of full 
                        visibility calculation [False]
        approx:         Kind of integration for visibilities (approx, numeric, analytic)
        mindatapoints:  if less valid datapoints in one baseline, file is rejected [3]
        onlypol1:       Only fits polarization 1 for split mode [False]
        flagtill:       Flag blue channels [2] 
        flagfrom:       Flag red channels [12]
        dontfit:        Number of telescope to flag
        dontfitbl:      Number of baseline to flag
        noS2:           If True ignores files where fiber offset is 0 [False]
        fixedBG:        Fit for background power law [False]
        fixedBH:        Fit for black hole power law [False]
        noBG:           Sets background flux ratio to 0 [True]
        fitopds:        Fit individual opds id equal 1 [0,0,0,0]
        specialpar:     Allows OPD for individual baseline [0,0,0,0,0,0]
        
        plot:           plot MCMC results [True]
        plotres:        plot fit result [True]
        writeresults:  Write fit results in file [True] 
        createpdf:      Creates a pdf with fit results and all plots [True] 
        writefitdiff:   Writes the difference of the mean fit vs data instead of redchi2
        returnPos:      Retuns fitted position
        """
        if self.resolution != 'LOW' and flagtill == 2 and flagfrom == 12:
            raise ValueError('Initial values for flagtill and flagfrom have to be changed if not low resolution')
        rad2as = 180 / np.pi * 3600
        self.fixedBG = fixedBG
        self.fixedBH = fixedBH
        self.noBG = noBG
        self.michistyle = michistyle
        self.approx = approx
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
        wave = self.wlSC
        
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
        if writeresults:
            txtfile = open(txtfilename, 'w')
            txtfile.write('# Results of Unary fit for %s \n' % self.name[stname:])
            txtfile.write('# Lines are: Best chi2, MCMC result, MCMC error -, MCMC error + \n')
            txtfile.write('# Rowes are: PC RA, PC DEC, alpha SgrA*, f BG, alpha BG \n')
            txtfile.write('# Parameter which are not fitted have 0.0 as error \n')
            txtfile.write('# MJD: %f \n' % MJD)
            txtfile.write('# OFFX: %f \n' % self.fiberOffX)
            txtfile.write('# OFFY: %f \n\n' % self.fiberOffY)

        self.wave = wave
        self.getDlambda()
        dlambda = self.dlambda

        # Initial guesses
        size = 5
        phase_center_RA = 0.01
        phase_center_DEC = 0.01
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
        
        if returnPos:
            returnList = []

        # Get data
        if self.polmode == 'SPLIT':
            if phaseinput is None:
                visphi_P = [self.visphiSC_P1, self.visphiSC_P2]
            else:
                visphi_P = phaseinput
            visphi_error_P = [self.visphierrSC_P1, self.visphierrSC_P2]
            visphi_flag_P = [self.visampflagSC_P1, self.visampflagSC_P2]
            
            ndit = np.shape(self.visampSC_P1)[0]//6
            if not bequiet:
                print('NDIT = %i' % ndit)
            polnom = 2
        elif self.polmode == 'COMBINED':
            visphi_P = [self.visphiSC]
            visphi_error_P = [self.visphierrSC]
            visphi_flag_P = [self.visampflagSC]
            
            ndit = np.shape(self.visampSC)[0]//6
            if not bequiet:
                print('NDIT = %i' % ndit)
            polnom = 1
            
        for dit in range(ndit):
            savetime = str(datetime.now()).replace('-', '')
            savetime = savetime.replace(' ', '-')
            savetime = savetime.replace(':', '')
            self.savetime = savetime
            if writeresults and ndit > 1:
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
                
                pdf.cell(40, 6, txt="Specialfit", ln=0, align="L", border=0)
                pdf.cell(40, 6, txt=str(specialfit), ln=0, align="L", border=0)
                pdf.cell(40, 6, txt="Specialfit BL", ln=0, align="L", border=0)
                pdf.cell(40, 6, txt=str(specialpar), ln=1, align="L", border=0)
                pdf.ln()

            if not bequiet:
                print('Run MCMC for DIT %i' % (dit+1))
            ditstart = dit*6
            ditstop = ditstart + 6


            if onlypol1:
                polnom = 1
            bothdofit = np.ones(polnom)            
            for idx in range(polnom):
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
                        
                    if returnPos:
                        returnList.append(percentiles[:,1])
                    
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
                if writeresults:
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
                        
                        if polnom == 2:
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
                            if polnom == 2:
                                text = '%s, redchi: %.2f (P1), %.2f (P2)' % (titles[pa], 
                                                                            redchi0, 
                                                                            redchi1)
                            else:
                                text = '%s, redchi: %.2f (P1)' % (titles[pa], redchi0)
                            pdf.cell(0, 10, txt=text, ln=1, align="C", border='B')
                            pdf.ln()
                            pdf.image(pdfimages0[pdfcout+pa], w=150)
                            if polnom == 2:
                                pdf.image(pdfimages1[pdfcout+pa], w=150)
                    
                    if not bequiet:
                        print('Save pdf as %s' % pdffilename)
                    pdf.output(pdffilename)
                else:
                    del pdf 
                files = glob.glob(savetime + '_pol?_?.png')
                for file in files:
                    os.remove(file)
        if writeresults:
            txtfile.close()
        if returnPos:
            return np.array(returnList)
        else:
            return 0
    
    
    
    
            
    def plotFitUnary(self, theta,  fitdata, u, v, idx=0, createpdf=False, uvind=False):
        rad2as = 180 / np.pi * 3600
        (visphi, visphi_error, visphi_flag) = fitdata
        visphi[np.isnan(visphi)] = 0
        wave = self.wlSC
        dlambda = self.dlambda
        if createpdf:
            savetime = self.savetime
        wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
        dlambda_model = np.zeros((6,len(wave_model)))
        for i in range(0,6):
            dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
            
        # Fit
        model_visphi_full = self.calc_vis_unary(theta, u, v, wave_model, dlambda_model)
        magu_as = self.spFrequAS
        
        u_as_model = np.zeros((len(u),len(wave_model)))
        v_as_model = np.zeros((len(v),len(wave_model)))
        for i in range(0,len(u)):
            u_as_model[i,:] = u[i]/(wave_model*1.e-6) / rad2as
            v_as_model[i,:] = v[i]/(wave_model*1.e-6) / rad2as
        magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)
        
        if uvind:
            gs = gridspec.GridSpec(1,2)
            axis = plt.subplot(gs[0,0])
            for i in range(0,6):
                plt.errorbar(u_as[i,:], visphi[i,:]*(1-visphi_flag[i]), 
                            visphi_error[i,:]*(1-visphi_flag[i]), label=self.baseline_labels[i],
                            color=self.colors_baseline[i], ls='', lw=1, alpha=0.5, capsize=0)
                plt.scatter(u_as[i,:], visphi[i,:]*(1-visphi_flag[i]),
                            color=self.colors_baseline[i], alpha=0.5)
                plt.plot(u_as_model[i,:], model_visphi_full[i,:],
                        color='k', zorder=100)
            plt.ylabel('visibility phase')
            plt.xlabel('U (1/arcsec)')     

            axis = plt.subplot(gs[0,1])
            for i in range(0,6):
                plt.errorbar(v_as[i,:], visphi[i,:]*(1-visphi_flag[i]), 
                            visphi_error[i,:]*(1-visphi_flag[i]), label=self.baseline_labels[i],
                            color=self.colors_baseline[i], ls='', lw=1, alpha=0.5, capsize=0)
                plt.scatter(v_as[i,:], visphi[i,:]*(1-visphi_flag[i]),
                            color=self.colors_baseline[i], alpha=0.5)
                plt.plot(v_as_model[i,:], model_visphi_full[i,:],
                        color='k', zorder=100)
            plt.xlabel('V (1/arcsec)')     
            plt.show()
        
        else:
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
            
