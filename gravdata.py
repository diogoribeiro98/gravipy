from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import scipy as sp
from scipy import signal, optimize, interpolate
from pkg_resources import resource_filename
from astropy.time import Time
from datetime import timedelta, datetime
from .phasecor import averaging

import sys
import os

try:
    from generalFunctions import *
    set_style('show')
except (NameError, ModuleNotFoundError):
    pass


color1 = '#C02F1D'
color2 = '#348ABD'
color3 = '#F26D21'
color4 = '#7A68A6'

def fiber_coupling(x):
    fiber_coup = np.exp(-1*(2*np.pi*np.sqrt(np.sum(x**2))/280)**2)
    return fiber_coup


def convert_date(date, mjd=False):
    t = Time(date)
    if mjd:
        return t.mjd
    t2 = Time('2000-01-01T12:00:00')
    date_decimal = (t.mjd - t2.mjd)/365.25+2000

    date = date.replace('T', ' ')
    date = date.split('.')[0]
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return date_decimal, date


def get_met(Volts, fc=False, removefc=True):
    V = np.array([np.convolve(Volts[:, i], np.ones(100), 'same') for i in range(80)]).T
    VC = V[:, 1::2] + 1j * V[:, ::2]
    
    if fc:
        VFCFT, VFCST = VC[:, 32:36], VC[:, 36:]
        VFCFT = (VFCFT) / abs(VFCFT)
        VFCST = (VFCST) / abs(VFCST)
        phaseFT = np.angle(VFCFT * np.conj(VFCFT.mean(axis=0)))
        phaseSC = np.angle(VFCST * np.conj(VFCST.mean(axis=0)))
        return phaseFT, phaseSC
    
    VCT = np.zeros((len(VC), 32), dtype=complex)
    if removefc:
        for i in range(8):
            VCT[:, 4 * i : 4 * (i + 1)] = (
                VC[:, 4 * i : 4 * (i + 1)] * np.conj(VC[:, 32 + i])[:, None]
            )
    else:
        VCT = VC[:,:-8]
    # Second np.convolve with time to gain in SNR (400DIT=800ms)
    VTEL = np.array([np.convolve(VCT[:, i], np.ones(150), "same") for i in range(32)]).T
    VTELFC = (VTEL[:, :16] * np.conj(VTEL[:, 16:])).reshape(-1,4,4)
    VTELFT = VTEL[:, :16].reshape(-1,4,4)
    VTELST = VTEL[:, 16:].reshape(-1,4,4)
    
    for i in range(4):
        for j in range(4):
            VTELFT[:,i,j]=np.convolve(VTELFT[:,i,j],np.ones(100),'same')
            VTELST[:,i,j]=np.convolve(VTELST[:,i,j],np.ones(100),'same')
    VTELFT = (VTELFT) / abs(VTELFT)
    VTELST = (VTELST) / abs(VTELST)
    phaseFT = np.unwrap(np.angle(VTELFT * np.conj(VTELFT.mean(axis=0))), axis=0)
    phaseSC = np.unwrap(np.angle(VTELST * np.conj(VTELST.mean(axis=0))), axis=0)
    rmsFT = np.std(phaseFT, axis=0)
    rmsSC = np.std(phaseSC, axis=0)
    phaseFT = np.angle(VTELFT * np.conj(VTELFT.mean(axis=0)))
    phaseSC = np.angle(VTELST * np.conj(VTELST.mean(axis=0)))
    return phaseFT, phaseSC, rmsFT, rmsSC


class GravData():
    def __init__(self, data, verbose=True, plot=False):
        """
        """
        self.name = data
        self.filename = os.path.basename(data)
        self.verbose = verbose
        self.colors_baseline = np.array(['k', 'darkblue', color4,
                                         color2, 'darkred', color1])
        self.colors_closure = np.array([color1, 'darkred', 'k', color2])

        poscatg = ['VIS_DUAL_SCI_RAW', 'VIS_SINGLE_SCI_RAW',
                   'VIS_SINGLE_CAL_RAW', 'VIS_DUAL_CAL_RAW',
                   'VIS_SINGLE_CALIBRATED', 'VIS_DUAL_CALIBRATED',
                   'SINGLE_SCI_VIS', 'SINGLE_SCI_VIS_CALIBRATED',
                   'DUAL_SCI_VIS', 'DUAL_SCI_VIS_CALIBRATED',
                   'SINGLE_CAL_VIS', 'DUAL_CAL_VIS', 'ASTROREDUCED',
                   'DUAL_SCI_P2VMRED']

        header = fits.open(self.name)[0].header
        date = header['DATE-OBS']

        self.header = header
        self.date = convert_date(date)
        self.raw = False

        if 'GRAV' not in header['INSTRUME']:
            raise ValueError('File seems to be not from GRAVITY')
        else:
            datatype = 'RAW' # default data type RAW
        if 'ESO PRO TYPE' in header:
            datatype = header['ESO PRO TYPE']
        if 'ESO PRO CATG' in header:
            datacatg = header['ESO PRO CATG']
            if datacatg not in poscatg:
                raise ValueError('filetype is %s, which is not supported'
                                 % datacatg)
        else:
            if self.verbose:
                print('Assume this is a raw file!')
            self.raw = True
            datacatg = 'RAW'

        self.datacatg = datacatg
        self.polmode = header['ESO INS POLA MODE']
        self.resolution = header['ESO INS SPEC RES']
        self.dit = header['ESO DET2 SEQ1 DIT']
        self.ndit = header['ESO DET2 NDIT']
        self.mjd = header['MJD-OBS']

        if 'P2VM' in self.datacatg:
            self.p2vm_file = True
        else:
            self.p2vm_file = False

        tel = fits.open(self.name)[0].header["TELESCOP"]
        if tel == 'ESO-VLTI-U1234':
            self.tel = 'UT'
        elif tel == 'ESO-VLTI-A1234':
            self.tel = 'AT'
        else:
            raise ValueError('Telescope not AT or UT, seomtehign '
                             'wrong with input data')

        # Get BL names
        if self.raw or self.p2vm_file:
            self.baseline_labels = np.array(["UT4-3", "UT4-2", "UT4-1",
                                            "UT3-2", "UT3-1", "UT2-1"])
            self.closure_labels = np.array(["UT4-3-2", "UT4-3-1",
                                            "UT4-2-1", "UT3-2-1"])
        else:
            baseline_labels = []
            closure_labels = []
            tel_name = fits.open(self.name)['OI_ARRAY'].data['TEL_NAME']
            sta_index = fits.open(self.name)['OI_ARRAY'].data['STA_INDEX']
            if self.polmode == 'SPLIT':
                vis_index = fits.open(self.name)['OI_VIS', 11].data['STA_INDEX']
                t3_index = fits.open(self.name)['OI_T3', 11].data['STA_INDEX']
            else:
                vis_index = fits.open(self.name)['OI_VIS', 10].data['STA_INDEX']
                t3_index = fits.open(self.name)['OI_T3', 10].data['STA_INDEX']
            for bl in range(6):
                t1 = np.where(sta_index == vis_index[bl, 0])[0][0]
                t2 = np.where(sta_index == vis_index[bl, 1])[0][0]
                baseline_labels.append(tel_name[t1] + '-' + tel_name[t2][2])
            for cl in range(4):
                t1 = np.where(sta_index == t3_index[cl, 0])[0][0]
                t2 = np.where(sta_index == t3_index[cl, 1])[0][0]
                t3 = np.where(sta_index == t3_index[cl, 2])[0][0]
                closure_labels.append(tel_name[t1] + '-'
                                      + tel_name[t2][2] + '-' + tel_name[t3][2])
            self.closure_labels = np.array(closure_labels)
            self.baseline_labels = np.array(baseline_labels)

        if self.verbose:
            print('Category: %s' % self.datacatg)
            print('Telescope: %s' % self.tel)
            print('Polarization: %s' % self.polmode)
            print('Resolution: %s' % self.resolution)
            print('DIT: %f' % self.dit)
            print('NDIT: %i' % self.ndit)

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

    def getIntdata(self, mode='SC', plot=False, plotTAmp=False, flag=False,
                   reload=False, ignore_tel=[]):
        """
        Reads out all interferometric data and saves it into the class:
        visamp, visphi, visphi2, closure amplitude and phase
        if plot it plots all data

        """
        if self.raw:
            raise ValueError('Input is a RAW file,',
                             'not usable for this function')
        if self.p2vm_file:
            raise ValueError('Input is a p2vmred file,',
                             'not usable for this function')

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
                    u_as[i, :] = (self.u[i] / (wave * 1.e-6)
                                  * np.pi / 180. / 3600.)  # 1/as
                    v_as[i, :] = (self.v[i] / (wave * 1.e-6)
                                  * np.pi / 180. / 3600.)  # 1/as
                self.spFrequAS = np.sqrt(u_as**2.+v_as**2.)

                # spatial frequency T3
                magu = np.sqrt(self.u**2.+self.v**2.)
                max_spf = np.zeros((len(magu)//6*4))
                for idx in range(len(magu)//6):
                    max_spf[0 + idx*4] = np.max(np.array([magu[0 + idx*6],
                                                          magu[3 + idx*6],
                                                          magu[1 + idx*6]]))
                    max_spf[1 + idx*4] = np.max(np.array([magu[0 + idx*6],
                                                          magu[4 + idx*6],
                                                          magu[2 + idx*6]]))
                    max_spf[2 + idx*4] = np.max(np.array([magu[1 + idx*6],
                                                          magu[5 + idx*6],
                                                          magu[2 + idx*6]]))
                    max_spf[3 + idx*4] = np.max(np.array([magu[3 + idx*6],
                                                          magu[5 + idx*6],
                                                          magu[4 + idx*6]]))

                self.max_spf = max_spf
                spFrequAS_T3 = np.zeros((len(max_spf),len(wave)))
                for idx in range(len(max_spf)):
                    spFrequAS_T3[idx] = (max_spf[idx]/(wave*1.e-6)
                                         * np.pi / 180. / 3600.)  # 1/as
                self.spFrequAS_T3 = spFrequAS_T3
                self.bispec_ind = np.array([[0,3,1],
                                            [0,4,2],
                                            [1,5,2],
                                            [3,5,4]])
                
                if not hasattr(self, 'visphiSC_P1') or reload:
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

                for t in ignore_tel:
                    for cdx, cl in enumerate(self.closure_labels):
                        if str(t) in cl:
                            self.t3flagSC_P1[cdx] = True
                            self.t3flagSC_P2[cdx] = True
                            self.t3ampflagSC_P1[cdx] = True
                            self.t3ampflagSC_P2[cdx] = True
                    for vdx, vi in enumerate(self.baseline_labels):
                        if str(t) in vi:
                            self.visampflagSC_P1[vdx] = True
                            self.visampflagSC_P2[vdx] = True
                            self.vis2flagSC_P1[vdx] = True
                            self.vis2flagSC_P2[vdx] = True
                            self.visphiflagSC_P1[vdx] = True
                            self.visphiflagSC_P2[vdx] = True

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
                        plt.errorbar(self.spFrequAS[idx,:],
                                     self.visampSC_P1[idx,:],
                                     self.visamperrSC_P1[idx,:],
                                     alpha=0.7, ms=4, lw=0.5, capsize=0,
                                     ls='', marker='o',
                                     color=self.colors_baseline[idx % 6])
                    for idx in range(len(self.vis2SC_P2)):
                        plt.errorbar(self.spFrequAS[idx,:],
                                     self.visampSC_P2[idx,:],
                                     self.visamperrSC_P2[idx,:],
                                     alpha=0.7, ms=4, lw=0.5, capsize=0,
                                     ls='', marker='D',
                                     color=self.colors_baseline[idx % 6])
                    plt.axhline(1, ls='--', lw=0.5)
                    plt.ylim(-0.0,1.1)
                    plt.ylabel('visibility amplitude')

                    axis = plt.subplot(gs[0,1])
                    for idx in range(len(self.vis2SC_P1)):
                        plt.errorbar(self.spFrequAS[idx,:],
                                     self.vis2SC_P1[idx,:],
                                     self.vis2errSC_P1[idx,:],
                                     alpha=0.7, ms=4, lw=0.5, capsize=0,
                                     ls='', marker='o',
                                     color=self.colors_baseline[idx % 6],
                                     label=self.baseline_labels[idx % 6])
                        if idx == 5:
                            plt.legend(frameon=True)
                    for idx in range(len(self.vis2SC_P2)):
                        plt.errorbar(self.spFrequAS[idx,:],
                                     self.vis2SC_P2[idx,:],
                                     self.vis2errSC_P2[idx,:],
                                     alpha=0.7, ms=4, lw=0.5, capsize=0,
                                     ls='', marker='D',
                                     color=self.colors_baseline[idx % 6])
                    plt.axhline(1, ls='--', lw=0.5)
                    plt.ylim(-0.0,1.1)
                    plt.ylabel('visibility squared')

                    axis = plt.subplot(gs[1,0])
                    for idx in range(len(self.t3SC_P2)):
                        plt.errorbar(self.spFrequAS_T3[idx,:],
                                     self.t3SC_P2[idx,:],
                                     self.t3errSC_P2[idx,:],
                                     alpha=0.7, ms=4, lw=0.5,
                                     capsize=0, marker='o', ls='',
                                     color=self.colors_closure[idx % 4],
                                     label=self.closure_labels[idx % 4])
                        if idx == 3:
                            plt.legend(frameon=True)
                    for idx in range(len(self.t3SC_P2)):
                        plt.errorbar(self.spFrequAS_T3[idx,:],
                                     self.t3SC_P1[idx,:],
                                     self.t3errSC_P1[idx,:],
                                     alpha=0.7, ms=4, lw=0.5, capsize=0,
                                     marker='D', ls='',
                                     color=self.colors_closure[idx % 4])
                    plt.axhline(0, ls='--', lw=0.5)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('closure phase (deg)')

                    axis = plt.subplot(gs[1, 1])
                    if plotTAmp:
                        for idx in range(len(self.t3SC_P2)):
                            plt.errorbar(self.spFrequAS_T3[idx,:],
                                         self.t3ampSC_P2[idx,:],
                                         self.t3amperrSC_P2[idx,:],
                                         marker='o', ls='',
                                         color=self.colors_closure[idx % 4])
                        for idx in range(len(self.t3SC_P2)):
                            plt.errorbar(self.spFrequAS_T3[idx,:],
                                         self.t3ampSC_P1[idx,:],
                                         self.t3amperrSC_P1[idx,:],
                                         marker='p', ls='',
                                         color=self.colors_closure[idx % 4])
                        plt.axhline(1, ls='--', lw=0.5)
                        plt.ylim(-0.0, 1.1)
                        plt.xlabel('spatial frequency (1/arcsec)')
                        plt.ylabel('closure amplitude')

                        axis = plt.subplot(gs[2, 1])

                    for idx in range(len(self.vis2SC_P1)):
                        plt.errorbar(self.spFrequAS[idx,:],
                                     self.visphiSC_P1[idx,:],
                                     self.visphierrSC_P1[idx,:],
                                     alpha=0.7, ms=4, lw=0.5, capsize=0,
                                     ls='', marker='o',
                                     color=self.colors_baseline[idx % 6])
                    for idx in range(len(self.vis2SC_P2)):
                        plt.errorbar(self.spFrequAS[idx,:],
                                     self.visphiSC_P2[idx,:],
                                     self.visphierrSC_P2[idx,:],
                                     alpha=0.7, ms=4, lw=0.5, capsize=0,
                                     ls='', marker='p',
                                     color=self.colors_baseline[idx % 6])
                    plt.axhline(0, ls='--', lw=0.5)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('visibility phase')
                    plt.show()

        if self.polmode =='COMBINED':
            if mode =='SC':
                self.u = fitsdata['OI_VIS', 10].data.field('UCOORD')
                self.v = fitsdata['OI_VIS', 10].data.field('VCOORD')

                # spatial frequency
                wave = self.wlSC
                self.wave = wave
                u_as = np.zeros((len(self.u),len(wave)))
                v_as = np.zeros((len(self.v),len(wave)))
                for i in range(0, len(self.u)):
                    u_as[i, :] = (self.u[i]/(wave*1.e-6)
                                  * np.pi / 180. / 3600.)  # 1/as
                    v_as[i, :] = (self.v[i]/(wave*1.e-6)
                                  * np.pi / 180. / 3600.)  # 1/as
                self.spFrequAS = np.sqrt(u_as**2.+v_as**2.)

                # spatial frequency T3
                magu = np.sqrt(self.u**2.+self.v**2.)
                max_spf = np.zeros(int(len(magu)/6*4))
                for idx in range(len(magu)//6):
                    max_spf[0 + idx*4] = np.max(np.array([magu[0 + idx*6],
                                                          magu[3 + idx*6],
                                                          magu[1 + idx*6]]))
                    max_spf[1 + idx*4] = np.max(np.array([magu[0 + idx*6],
                                                          magu[4 + idx*6],
                                                          magu[2 + idx*6]]))
                    max_spf[2 + idx*4] = np.max(np.array([magu[1 + idx*6],
                                                          magu[5 + idx*6],
                                                          magu[2 + idx*6]]))
                    max_spf[3 + idx*4] = np.max(np.array([magu[3 + idx*6],
                                                          magu[5 + idx*6],
                                                          magu[4 + idx*6]]))
                self.max_spf = max_spf
                spFrequAS_T3 = np.zeros((len(max_spf),len(wave)))
                for idx in range(len(max_spf)):
                    spFrequAS_T3[idx] = (max_spf[idx]/(wave*1.e-6)
                                         * np.pi / 180. / 3600.)  # 1/as
                self.spFrequAS_T3 = spFrequAS_T3
                self.bispec_ind = np.array([[0, 3, 1],
                                            [0, 4, 2],
                                            [1, 5, 2],
                                            [3, 5, 4]])

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

                for t in ignore_tel:
                    for cdx, cl in enumerate(self.closure_labels):
                        if str(t) in cl:
                            self.t3flagSC[cdx] = True
                            self.t3ampflagSC[cdx] = True
                    for vdx, vi in enumerate(self.baseline_labels):
                        if str(t) in vi:
                            self.visampflagSC[vdx] = True
                            self.vis2flagSC[vdx] = True
                            self.visphiflagSC[vdx] = True

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
                        gs = gridspec.GridSpec(3, 2)
                        plt.figure(figsize=(15, 15))
                    else:
                        gs = gridspec.GridSpec(2, 2)
                        plt.figure(figsize=(15, 12))
                    axis = plt.subplot(gs[0, 0])
                    for idx in range(len(self.vis2SC)):
                        plt.errorbar(self.spFrequAS[idx],
                                     self.visampSC[idx],
                                     self.visamperrSC[idx],
                                     ls='', marker='o',
                                     color=self.colors_baseline[idx % 6])
                    plt.axhline(1, ls='--', lw=0.5)
                    plt.ylim(-0.0, 1.1)
                    plt.ylabel('visibility amplitude')

                    axis = plt.subplot(gs[0,1])
                    for idx in range(len(self.vis2SC)):
                        plt.errorbar(self.spFrequAS[idx],
                                     self.vis2SC[idx],
                                     self.vis2errSC[idx],
                                     ls='', marker='o',
                                     color=self.colors_baseline[idx % 6])
                    plt.axhline(1, ls='--', lw=0.5)
                    plt.ylim(-0.0, 1.1)
                    plt.ylabel('visibility squared')


                    axis = plt.subplot(gs[1,0])
                    for idx in range(len(self.t3SC)):
                        plt.errorbar(self.spFrequAS_T3[idx],
                                     self.t3SC[idx],
                                     self.t3errSC[idx,:],
                                     marker='o', ls='',
                                     color=self.colors_closure[idx % 4])
                    plt.axhline(0, ls='--', lw=0.5)
                    plt.xlabel('spatial frequency (1/arcsec)')
                    plt.ylabel('closure phase (deg)')

                    axis = plt.subplot(gs[1, 1])
                    if plotTAmp:
                        for idx in range(len(self.t3SC)):
                            plt.errorbar(self.spFrequAS_T3[idx],
                                         self.t3ampSC[idx],
                                         self.t3amperrSC[idx],
                                         marker='o', ls='',
                                         color=self.colors_closure[idx%4])
                        plt.axhline(1, ls='--', lw=0.5)
                        plt.ylim(-0.0, 1.1)
                        plt.xlabel('spatial frequency (1/arcsec)')
                        plt.ylabel('closure amplitude')

                        axis = plt.subplot(gs[2, 1])
                    for idx in range(len(self.vis2SC)):
                        plt.errorbar(self.spFrequAS[idx],
                                     self.visphiSC[idx],
                                     self.visphierrSC[idx],
                                     ls='', marker='o',
                                     color=self.colors_baseline[idx % 6])
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
                    print('High mode, or weird mode')
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

    def av_phases(self, phases, axis=0):
        phases = np.exp(1j*np.radians(phases))
        phases = (np.nanmean(np.real(phases), axis=axis)
                  + 1j*np.nanmean(np.imag(phases), axis=axis))
        phases = np.angle(phases, deg=True)
        return phases

    def calibrate_phi(self, calibrator):
        if not hasattr(self, 'visphiSC_P1'):
            self.getIntdata()
        c = fits.open(calibrator)
        c_channel = len(c['OI_WAVELENGTH', 11].data['EFF_WAVE'])
        if c_channel != self.channel:
            raise ValueError('Calibrator has different number '
                             'of spectral channels')
        cP1 = c['OI_VIS', 11].data['VISPHI']
        cP2 = c['OI_VIS', 12].data['VISPHI']
        cf1 = c['OI_VIS', 11].data['FLAG']
        cf2 = c['OI_VIS', 12].data['FLAG']
        cP1[cf1] = np.nan
        cP2[cf2] = np.nan
        cP1 = self.av_phases(cP1.reshape(-1, 6, self.channel))[np.newaxis,:,:]
        cP2 = self.av_phases(cP2.reshape(-1, 6, self.channel))[np.newaxis,:,:]

        sP1 = self.visphiSC_P1.reshape(-1, 6, self.channel)
        sP2 = self.visphiSC_P2.reshape(-1, 6, self.channel)

        self.visphiSC_P1 = np.angle(np.exp(1j*np.radians(sP1))
                                    / np.exp(1j*np.radians(cP1)), deg=True)
        self.visphiSC_P2 = np.angle(np.exp(1j*np.radians(sP2))
                                    / np.exp(1j*np.radians(cP2)), deg=True)

        if self.visphiSC_P1.shape[0] == 1:
            self.visphiSC_P1 = self.visphiSC_P1[0]
            self.visphiSC_P2 = self.visphiSC_P2[0]
        else:
            self.visphiSC_P1 = self.visphiSC_P1.reshape(-1,self.channel)
            self.visphiSC_P2 = self.visphiSC_P2.reshape(-1,self.channel)

class GravNight():
    def __init__(self, file_list, verbose=True):
        """
        GravNight - the long awaited full night fit class
        """
        self.file_list = file_list
        self.verbose = verbose
        self.colors_baseline = np.array(['k', 'darkblue', color4, 
                                         color2, 'darkred', color1])
        self.colors_closure = np.array([color1, 'darkred', 'k', color2])
        self.colors_tel = np.array([color1, 'darkred', 'k', color2])
        self.get_files()

    def get_files(self):
        self.datalist = []
        self.headerlist = []

        for num, fi in enumerate(self.file_list):
            self.datalist.append(GravData(fi, verbose=False))
            self.headerlist.append(fits.open(fi)[0].header)

        _catg = [i.datacatg for i in self.datalist]
        if _catg.count(_catg[0]) == len(_catg):
            self.datacatg = _catg[0]
        else:
            print(_catg)
            raise ValueError('Not all input data from same category')

        _tel = [i.tel for i in self.datalist]
        if _tel.count(_tel[0]) == len(_tel):
            self.tel = _tel[0]
        else:
            print(_tel)
            raise ValueError('Not all input data from same tel')

        _pol = [i.polmode for i in self.datalist]
        if _pol.count(_pol[0]) == len(_pol):
            self.polmode = _pol[0]
        else:
            print(_pol)
            raise ValueError('Not all input data from same polmode')

        _res = [i.resolution for i in self.datalist]
        if _res.count(_res[0]) == len(_res):
            self.resolution = _res[0]
        else:
            print(_res)
            raise ValueError('Not all input data from same resolution')

        _dit = [i.dit for i in self.datalist]
        if _dit.count(_dit[0]) == len(_dit):
            self.dit = _dit[0]
        else:
            print(_dit)
            raise ValueError('Not all input data from same dit')

        _ndit = [i.ndit for i in self.datalist]
        if _ndit.count(_ndit[0]) == len(_ndit):
            self.ndit = _ndit[0]
        else:
            print(_ndit)
            raise ValueError('Not all input data from same ndit')

        if self.verbose:
            print('Data loaded as:')
            print('Category: %s' % self.datacatg)
            print('Telescope: %s' % self.tel)
            print('Polarization: %s' % self.polmode)
            print('Resolution: %s' % self.resolution)
            print('DIT: %f' % self.dit)
            print('NDIT: %i' % self.ndit)

        self.mjd = [i.mjd for i in self.datalist]
        self.mjd0 = np.min(np.array(self.mjd))
        self.files = [i.name for i in self.datalist]

    def getIntdata(self, mode='SC', plot=False, plotTAmp=False, flag=False,
                   ignore_tel=[]):
        for data in self.datalist:
            data.getIntdata(mode=mode, plot=plot, plotTAmp=plotTAmp, flag=flag,
                             ignore_tel=ignore_tel)

    def getTime(self):
        files = self.files
        MJD = np.array([]).reshape(0,4)
        for fdx, file in enumerate(files):
            d = fits.open(file)['OI_FLUX'].data
            _MJD0 = fits.open(file)[0].header['MJD-OBS']
            MJD = np.concatenate((MJD, d['TIME'].reshape(-1, 4)/1e6/3600/24
                                  + _MJD0))
        MJD = (MJD - self.mjd0)*24*60
        self.time = MJD

    def getMetdata(self, plot=False):
        if 'P2VM' not in self.datacatg:
            raise ValueError('Only available for p2vmred files')
        files = self.files
        if self.polmode == 'SPLIT':
            fitnum = 11
        else:
            fitnum = 10

        MJD = np.array([]).reshape(0,4)
        OPD_FC = np.array([]).reshape(0,4)
        OPD_FC_CORR = np.array([]).reshape(0,4)
        OPD_TEL = np.array([]).reshape(0,4,4)
        OPD_TEL_CORR = np.array([]).reshape(0,4,4)
        OPD_TELFC_CORR = np.array([]).reshape(0,4,4)
        PHA_TELFC_CORR = np.array([]).reshape(0,4,4)
        OPD_TELFC_MCORR = np.array([]).reshape(0,4)
        E_U = np.array([]).reshape(0,4,3)
        E_V = np.array([]).reshape(0,4,3)

        for fdx, file in enumerate(files):
            d = fits.open(file)['OI_VIS_MET'].data
            _MJD0 = fits.open(file)[0].header['MJD-OBS']
            MJD = np.concatenate((MJD,
                                  d['TIME'].reshape(-1, 4)/1e6/3600/24 + _MJD0))
            OPD_FC = np.concatenate((OPD_FC, d['OPD_FC'].reshape(-1, 4)*1e6))
            OPD_FC_CORR = np.concatenate((OPD_FC_CORR, 
                                          d['OPD_FC_CORR'].reshape(-1, 4)*1e6))
            OPD_TELFC_MCORR = np.concatenate((OPD_TELFC_MCORR, 
                                              d['OPD_TELFC_MCORR'].reshape(-1, 4)*1e6))

            E_U = np.concatenate((E_U, d['E_U'].reshape(-1, 4, 3)))
            E_V = np.concatenate((E_V, d['E_V'].reshape(-1, 4, 3)))

            OPD_TEL = np.concatenate((OPD_TEL,
                                      d['OPD_TEL'].reshape(-1, 4, 4)*1e6))
            OPD_TEL_CORR = np.concatenate((OPD_TEL_CORR,
                                           d['OPD_TEL_CORR'].reshape(-1, 4, 4)
                                           * 1e6))
            OPD_TELFC_CORR = np.concatenate((OPD_TELFC_CORR,
                                             d['OPD_TELFC_CORR'].reshape(-1, 4, 4)
                                             * 1e6))
            PHA_TELFC_CORR = np.concatenate((PHA_TELFC_CORR,
                                             d['PHASE_TELFC_CORR'].reshape(-1, 4, 4)
                                             ))

        MJD = (MJD - self.mjd0)*24*60
        self.time = MJD
        self.opd_fc = OPD_FC
        self.opd_fc_corr = OPD_FC_CORR
        self.opd_telfc_mcorr = OPD_TELFC_MCORR
        self.e_u = E_U
        self.e_v = E_V
        self.opd_tel = OPD_TEL
        self.opd_tel_corr = OPD_TEL_CORR
        self.opd_telfc_corr = OPD_TELFC_CORR
        self.pha_telfc_corr = PHA_TELFC_CORR
        self.mjd_files = []
        self.ut_files = []
        self.lst_files = []
        for idx, file in enumerate(files):
            d = fits.open(file)
            self.mjd_files.append(d['OI_VIS', fitnum].data['MJD'][0])
            a = file.find('GRAVI.20')
            self.ut_files.append(file[a+17:a+22])
            self.lst_files.append(d[0].header['LST'])
        self.t_files = (np.array(self.mjd_files)-self.mjd0)*24*60

        if plot:
            av = 100
            maxval = []
            for tel in range(4):
                for dio in range(4):
                    maxval.append(np.max(np.abs(averaging(OPD_TEL[:, tel, dio]-np.mean(OPD_TEL[:, tel, dio]), av))))
            maxval = np.max(maxval)*1.2

            gs = gridspec.GridSpec(4,4, wspace=0.05, hspace=0.05)
            plt.figure(figsize=(7,7))
            for tel in range(4):
                for dio in range(4):
                    ax = plt.subplot(gs[tel,dio])
                    plt.plot(averaging(MJD[:, tel], av), averaging(OPD_TEL[:, tel, dio]-np.mean(OPD_TEL[:, tel, dio]), av), 
                            ls='', marker='.', label='UT%i\nDiode %i' % ((4-tel), dio), color=self.colors_tel[tel])
                    for m in range(len(self.t_files)):
                        plt.axvline(self.t_files[m], ls='--', lw=0.2, color='grey')
                        if tel == 0 and dio == 0:
                            plt.text(self.t_files[m]+0.5, -maxval*0.9, self.ut_files[m], rotation=90, fontsize=5)
                    plt.legend(loc=2)
                    plt.ylim(-maxval,maxval)
                    if tel != 3:
                        ax.set_xticklabels([])
                    else:
                        plt.xlabel('Time [mins]', fontsize=8)
                    if dio != 0:
                        ax.set_yticklabels([])
                    else:
                        plt.ylabel('OPD_TEL [$\mu$m]', fontsize=8)
            plt.show()


            maxval = []
            for tel in range(4):
                maxval.append(np.max(np.abs(averaging(OPD_TELFC_MCORR[:, tel]
                                                      - np.mean(OPD_TELFC_MCORR[:, tel]),
                                                      av))))
            maxval = np.max(maxval)*1.2
            gs = gridspec.GridSpec(1, 4, wspace=0.05, hspace=0.05)
            plt.figure(figsize=(7,2))
            for tel in range(4):
                ax = plt.subplot(gs[0,tel])
                plt.plot(averaging(MJD[:, tel], av),
                         averaging(OPD_TELFC_MCORR[:, tel]
                                   - np.mean(OPD_TELFC_MCORR[:, tel]), av),
                         ls='', marker='.', label='UT%i' % (4-tel),
                         color=self.colors_tel[tel])
                for m in range(len(self.t_files)):
                    plt.axvline(self.t_files[m], ls='--', lw=0.2, color='grey')
                    if tel == 0:
                        plt.text(self.t_files[m]+0.5, -maxval*0.9,
                                 self.ut_files[m], rotation=90, fontsize=5)
                plt.legend(loc=2)
                plt.ylim(-maxval, maxval)
                plt.xlabel('Time [mins]', fontsize=8)
                if tel != 0:
                    ax.set_yticklabels([])
                else:
                    plt.ylabel('OPD_TELFC_MCORR \n[$\mu$m]', fontsize=8)
            plt.show()

    def getAcqdata(self, plot=False):
        if 'P2VM' not in self.datacatg:
            raise ValueError('Only available for p2vmred files')
        files = self.files
        if self.polmode == 'SPLIT':
            fitnum = 11
        else:
            fitnum = 10

        MJD = np.array([]).reshape(0, 4)
        PUPIL_X = np.array([]).reshape(0, 4)
        PUPIL_Y = np.array([]).reshape(0, 4)
        PUPIL_Z = np.array([]).reshape(0, 4)

        for fdx, file in enumerate(files):
            d = fits.open(file)['OI_VIS_ACQ'].data
            _MJD0 = fits.open(file)[0].header['MJD-OBS']
            MJD = np.concatenate((MJD,
                                  d['TIME'].reshape(-1, 4)/1e6/3600/24 + _MJD0))
            PUPIL_X = np.concatenate((PUPIL_X, d['PUPIL_X'].reshape(-1, 4)))
            PUPIL_Y = np.concatenate((PUPIL_Y, d['PUPIL_X'].reshape(-1, 4)))
            PUPIL_Z = np.concatenate((PUPIL_Z, d['PUPIL_X'].reshape(-1, 4)))

        MJD = (MJD - self.mjd0)*24*60
        self.acqtime = MJD
        self.pupil = np.array([PUPIL_X, PUPIL_Y, PUPIL_Z])
        self.pupil[self.pupil == 0] = np.nan
        for idx, file in enumerate(files):
            d = fits.open(file)
            self.mjd_files.append(d['OI_VIS', fitnum].data['MJD'][0])
            a = file.find('GRAVI.20')
            self.ut_files.append(file[a+17:a+22])
            self.lst_files.append(d[0].header['LST'])
        self.t_files = (np.array(self.mjd_files)-self.mjd0)*24*60

        if plot:
            maxval = np.nanmax(np.abs(self.pupil))*1.2
            pup_name = ['X', 'Y', 'Z']

            gs = gridspec.GridSpec(3, 4, wspace=0.05, hspace=0.05)
            plt.figure(figsize=(7, 7))
            for pup in range(3):
                for tel in range(4):
                    ax = plt.subplot(gs[pup, tel])
                    plt.plot(self.acqtime[:, tel], self.pupil[pup, :, tel],
                             ls='', marker='.',  markersize=1,
                             color=self.colors_tel[tel])
                    for m in range(len(self.t_files)):
                        plt.axvline(self.t_files[m], ls='--', lw=0.2,
                                    color='grey')
                        if tel == 0 and pup == 0:
                            plt.text(self.t_files[m]+0.5, -maxval*0.9,
                                     self.ut_files[m], rotation=90, fontsize=5)
                    plt.ylim(-maxval, maxval)
                    plt.axhline(0, ls='--', lw=1, zorder=0, color='grey')
                    if pup == 0:
                        plt.title('UT %i' % (4-tel), fontsize=8)
                    if pup != 2:
                        ax.set_xticklabels([])
                    else:
                        plt.xlabel('Time [mins]', fontsize=8)
                    if tel != 0:
                        ax.set_yticklabels([])
                    else:
                        plt.ylabel('PUPIL %s\n[pix]' % pup_name[pup],
                                   fontsize=8)
            plt.show()

    def getFainttimer(self):
        files = self.files
        onv = np.array([])
        ofv = np.array([])

        for file in files:
            h = fits.open(file)[0].header
            if h['ESO INS MET MODE'] != 'FAINT':
                raise ValueError('Metmode is not faint')
            rate1 = h['ESO INS ANLO3 RATE1']/60
            rate2 = h['ESO INS ANLO3 RATE2']/60
            repe1 = h['ESO INS ANLO3 REPEAT1']
            repe2 = h['ESO INS ANLO3 REPEAT2']
            time1 = h['ESO INS ANLO3 TIMER1']
            time2 = h['ESO INS ANLO3 TIMER2']
            # volt1 = h['ESO INS ANLO3 VOLTAGE1']
            # volt2 = h['ESO INS ANLO3 VOLTAGE2']

            mt1 = ((time1 / 86400.0) + 2440587.5 - 2400000.5 - self.mjd0)*24*60
            mt2 = ((time2 / 86400.0) + 2440587.5 - 2400000.5 - self.mjd0)*24*60

            onv = np.concatenate((onv, np.linspace(mt1, mt1+rate1*(repe1-1),
                                                   repe1)))
            ofv = np.concatenate((ofv, np.linspace(mt2, mt2+rate2*(repe2-1),
                                                   repe2)))

        try:
            self.onv = np.concatenate((onv, np.array([self.time[-1, 0]])))
            self.ofv = np.concatenate((np.array([self.time[0, 0]]), ofv))
        except AttributeError:
            try:
                self.getMetdata()
                self.onv = np.concatenate((onv, np.array([self.time[-1, 0]])))
                self.ofv = np.concatenate((np.array([self.time[0, 0]]), ofv))
            except ValueError:
                self.ofv = np.concatenate((np.array([0]), ofv))
                self.onv = np.concatenate((onv, np.array([ofv[-1] + ofv[0]])))

        # for num, fi in enumerate(self.file_list):
        # 
        #     if num == 0:
        #         type1_ = fits.getheader(fi)["ESO INS SOBJ X"]
        #         type2_ = fits.getheader(fi)["ESO INS SOBJ OFFX"]
        #         self.gravData_list.append(GravData(fi, verbose=self.verbose))
        #     else:
        #         if type1_ != fits.getheader(fi)["ESO INS SOBJ X"]:
        #             raise ValueError("all files need to be the same, but ", fits.getheader(fi)["ESO INS SOBJ X"], " is different from first file: ", type1_)
        #         if type2_ != fits.getheader(fi)["ESO INS SOBJ OFFX"]:
        #             raise ValueError("all files need to be the same, but ", fits.getheader(fi)["ESO INS SOBJ OFFX"], " is different from first file: ", type2_)
        #         self.gravData_list.append(GravData(fi, verbose=self.verbose))






