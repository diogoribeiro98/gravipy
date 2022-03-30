from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import emcee
import corner
from multiprocessing import Pool
from fpdf import FPDF
from PIL import Image
from scipy import signal, interpolate
import math
import mpmath
from pkg_resources import resource_filename
from numba import njit, prange
from datetime import datetime
import multiprocessing
import os
import pandas as pd
from joblib import Parallel, delayed

try:
    from numba import jit
except ModuleNotFoundError:
    print("can't import numba, please install it via pip or conda")

from .gravdata import *

try:
    from generalFunctions import *
    set_style('show')
except (NameError, ModuleNotFoundError):
    pass

import colorsys
import matplotlib.colors as mc


def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


color1 = '#C02F1D'
color2 = '#348ABD'
color3 = '#F26D21'
color4 = '#7A68A6'


# some numba functions for future!
# @njit('float64(float64[:], float64[:])', fastmath=False)
# def nb_trapz(y, x):
#     sz = y.shape[0]
#     res = 0
#     for i in range(sz-1):
#         res = res + (y[i+1]+y[i])*(x[i+1]-x[i])*0.5
#     return res
#
# @njit('float64[:](float64[:,:], float64[:], int64)', fastmath=False)
# def nb_trapz2d_ax(z, xy, axis):
#
#     sz1, sz2 = z.shape[0], z.shape[1]
#
#     if axis == 0:
#         res = np.empty(sz2)
#         for j in prange(sz2):
#             res[j] = nb_trapz(z[:,j], xy)
#         return res
#     elif axis == 1:
#         res = np.empty(sz1)
#         for i in prange(sz1):
#             res[i] = nb_trapz(z[i,:], xy)
#         return res
#     else:
#         raise ValueError

# @jit(nopython=True)
# def mathfunc_real(values, dt):
#     print(values.shape)
#     return nb_trapz2d_ax(np.real(values), dt, 0)

# @jit(nopython=True)
# def mathfunc_imag(values, dt):
#     print(values.shape)
#     return nb_trapz2d_ax(np.imag(values), dt, 0)

def mathfunc_real(values, dt):
    return np.trapz(np.real(values), dx=dt, axis=0)


def mathfunc_imag(values, dt):
    return np.trapz(np.imag(values), dx=dt, axis=0)


def complex_quadrature_num(func, a, b, theta, nsteps=int(1e2)):
    t = np.logspace(np.log10(a), np.log10(b), nsteps)
    dt = np.diff(t, axis=0)
    values = func(t, *theta)
    real_integral = mathfunc_real(values, dt)
    imag_integral = mathfunc_imag(values, dt)
    return real_integral + 1j*imag_integral


def procrustes(a, target, padval=0):
    try:
        if len(target) != a.ndim:
            raise TypeError('Target shape must have the same number of dimensions as the input')
    except TypeError:
        raise TypeError('Target must be array-like')
    try:
        # Get array in the right size to use
        b = np.ones(target,a.dtype)*padval
    except TypeError:
        raise TypeError('Pad value must be numeric')
    except ValueError:
        raise ValueError('Pad value must be scalar')

    aind = [slice(None, None)]*a.ndim
    bind = [slice(None, None)]*a.ndim

    for dd in range(a.ndim):
        if a.shape[dd] > target[dd]:
            diff = (a.shape[dd]-target[dd])/2.
            aind[dd] = slice(int(np.floor(diff)),
                             int(a.shape[dd]-np.ceil(diff)))
        elif a.shape[dd] < target[dd]:
            diff = (target[dd]-a.shape[dd])/2.
            bind[dd] = slice(int(np.floor(diff)),
                             int(target[dd]-np.ceil(diff)))
    b[bind] = a[aind]
    return b


class GravPhaseMaps():
    def createPhasemaps(self, nthreads=1, smooth=10, plot=True, datayear=2019):
        if datayear == 2019:
            zerfile = 'phasemap_zernike_20200918_diff_2019data.npy'
        elif datayear == 2020:
            zerfile = 'phasemap_zernike_20200922_diff_2020data.npy'
        else:
            raise ValueError('Datayear has to be 2019 or 2020')
        print('Used file: %s' % zerfile)

        def phase_screen(A00, A1m1, A1p1, A2m2, A2p2, A20, A3m1, A3p1, A3m3,
                         A3p3, A4m2, A4p2, A4m4, A4p4, A40,  A5m1, A5p1, A5m3,
                         A5p3, A5m5, A5p5, A6m6, A6p6, A6m4, A6p4, A6m2, A6p2,
                         A60, B1m1, B1p1, B20, B2m2, B2p2,
                         lam0=2.2, MFR=0.6308, stopB=8.0, stopS=0.96,
                         d1=8.0, dalpha=1., totN=1024, amax=100):
            """
            Simulate phase screens taking into account static aberrations.

            Parameters:
            -----------
            * Static aberrations in the pupil plane are described by 
            * low-order Zernicke polynomials
            * Their amplitudes are in units of micro-meter

            00: A00  (float) : piston
            01: A1m1 (float) : vertical tilt
            02: A1p1 (float) : horizontal tilt
            03: A2m2 (float) : vertical astigmatism
            04: A2p2 (float) : horizontal astigmatism
            05: A20  (float) : defocuss
            06: A3m1 (float) : vertical coma
            07: A3p1 (float) : horizontal coma
            08: A3m3 (float) : vertical trefoil
            09: A3p3 (float) : oblique trefoil
            10: A4m2 (float) : oblique secondary astigmatism
            11: A4p2 (float) : vertical secondary astigmatism
            12: A4m4 (float) : oblique quadrafoil
            13: A4p4 (float) : vertical quadrafoil
            14: A40  (float) : primary spherical
            15: A5m1
            16: A5p1
            17: A5m3
            18: A5p3
            19: A5m5
            20: A5p5
            21: A6m6
            22: A6p6
            23: A6m4
            24: A6p4
            25: A6m2
            26: A6p2
            27: A60

            * Static aberrations in the focal plane
            B1m1 (float) : missplacement of the fiber mode in u1-direction in
                           meters / Zernike coefficient if 
                           coefficients > B20 != 0
            B1p1 (float) : missplacement of the fiber mode in u2-direction in
                           meters / Zernike coefficient if 
                           coefficients > B20 != 0
            B20  (float) : defocuss
            B2m2 (float) : vertical astigmatism
            B2p2 (float) : horizontal astigmatism

            * optical system
            MFR (float)   : sigma of fiber mode profile in units of dish radius
            stopB (float) : outer stop diameter in meters
            stopS (float) : inner stop diameter in meters

            * further parameters specify the output grid
            dalpha (float) : pixel width in the imaging plane in mas
            totN   (float) : total number of pixels in the pupil plane
            lam0   (float) : wavelength at which the phase screen is computed in
                             micro-meter
            d1     (float) : telescope to normalize Zernike RMS in m 
                             (UT=8.0, AT=1.82)
            amax   (float) : maximum off-axis distance in the maps returned
            """

            # --- coordinate scaling --- #
            lam0 = lam0*1e-6
            mas = 1.e-3 * (2.*np.pi/360) * 1./3600
            ext = totN*d1/lam0*mas*dalpha*dalpha
            du = dalpha/ext*d1/lam0

            # --- coordinates --- #
            ii = np.arange(totN) - (totN/2)
            ii = np.fft.fftshift(ii)

            # image plane
            a1, a2 = np.meshgrid(ii*dalpha, ii*dalpha)
            aa = np.sqrt(a1*a1 + a2*a2)

            # pupil plane
            u1, u2 = np.meshgrid(ii*du*lam0, ii*du*lam0)
            r = np.sqrt( u1*u1 + u2*u2 )
            t = np.angle(u1 + 1j*u2)

            # --- cut our central part --- #
            hmapN = int(amax/dalpha)
            cc = slice(int(totN/2)-hmapN, int(totN/2)+hmapN+1)
            if 2*hmapN > totN:
                print('Requested map sizes too large')
                return False

            # --- pupil function --- #
            pupil = r<(stopB/2.)
            if stopS > 0.:
                pupil = np.logical_and(r < (stopB/2.), r > (stopS/2.))

            # --- fiber profile --- #
            fiber = np.exp(-0.5*(r/(MFR*d1/2.))**2)
            if B1m1 != 0 or B1p1 != 0:
                fiber = np.exp(-0.5*((u1-B1m1)**2 +
                                     (u2-B1p1)**2)/(MFR*d1/2.)**2)

            # for higher-order focal plane aberrations we need to compute the 
            # fourier transform explicitly
            if np.any([B20, B2m2, B2p2] != 0):
                sigma_fib = lam0/d1/np.pi/MFR/mas
                sigma_ref = 2.2e-6/d1/np.pi/MFR/mas
                zernike = 0
                zernike += B1m1*2*(aa/sigma_ref)*np.sin(t)
                zernike += B1p1*2*(aa/sigma_ref)*np.cos(t)
                zernike += B20*np.sqrt(3.)*(2.*(aa/sigma_ref)**2 - 1)
                zernike += B2m2*np.sqrt(6.)*(aa/sigma_ref)**2*np.sin(2.*t)
                zernike += B2p2*np.sqrt(6.)*(aa/sigma_ref)**2*np.cos(2.*t)

                fiber = (np.exp(-0.5*(aa/sigma_fib)**2)
                         * np.exp(2.*np.pi/lam0*1j*zernike*1e-6))
                fiber = np.fft.fft2(fiber)

            # --- phase screens (pupil plane) --- #
            zernike = A00
            zernike += A1m1*2*(2.*r/d1)*np.sin(t)
            zernike += A1p1*2*(2.*r/d1)*np.cos(t)
            zernike += A2m2*np.sqrt(6.)*(2.*r/d1)**2*np.sin(2.*t)
            zernike += A2p2*np.sqrt(6.)*(2.*r/d1)**2*np.cos(2.*t)
            zernike += A20*np.sqrt(3.)*(2.*(2.*r/d1)**2 - 1)
            zernike += A3m1*np.sqrt(8.)*(3.*(2.*r/d1)**3 - 2.*(2.*r/d1))*np.sin(t)
            zernike += A3p1*np.sqrt(8.)*(3.*(2.*r/d1)**3 - 2.*(2.*r/d1))*np.cos(t)
            zernike += A3m3*np.sqrt(8.)*(2.*r/d1)**3*np.sin(3.*t)
            zernike += A3p3*np.sqrt(8.)*(2.*r/d1)**3*np.cos(3.*t)
            zernike += A4m2*np.sqrt(10.)*(2.*r/d1)**4*np.sin(4.*t)
            zernike += A4p2*np.sqrt(10.)*(2.*r/d1)**4*np.cos(4.*t)
            zernike += A4m4*np.sqrt(10.)*(4.*(2.*r/d1)**4 -3.*(2.*r/d1)**2)*np.sin(2.*t)
            zernike += A4p4*np.sqrt(10.)*(4.*(2.*r/d1)**4 -3.*(2.*r/d1)**2)*np.cos(2.*t)
            zernike += A40*np.sqrt(5.)*(6.*(2.*r/d1)**4 - 6.*(2.*r/d1)**2 + 1)
            zernike += A5m1*2.*np.sqrt(3.)*(10*(2.*r/d1)**5 - 12*(2.*r/d1)**3 + 3.*2.*r/d1)*np.sin(t)
            zernike += A5p1*2.*np.sqrt(3.)*(10*(2.*r/d1)**5 - 12*(2.*r/d1)**3 + 3.*2.*r/d1)*np.cos(t)
            zernike += A5m3*2.*np.sqrt(3.)*(5.*(2.*r/d1)**5 - 4.*(2.*r/d1)**3)*np.sin(3.*t)
            zernike += A5p3*2.*np.sqrt(3.)*(5.*(2.*r/d1)**5 - 4.*(2.*r/d1)**3)*np.cos(3.*t)
            zernike += A5m5*2.*np.sqrt(3.)*(2.*r/d1)**5*np.sin(5*t)
            zernike += A5p5*2.*np.sqrt(3.)*(2.*r/d1)**5*np.cos(5*t)
            zernike += A6m6*np.sqrt(14.)*(2.*r/d1)**6*np.sin(6.*t)
            zernike += A6p6*np.sqrt(14.)*(2.*r/d1)**6*np.cos(6.*t)
            zernike += A6m4*np.sqrt(14.)*(6.*(2.*r/d1)**6 - 5.*(2.*r/d1)**4)*np.sin(4.*t)
            zernike += A6p4*np.sqrt(14.)*(6.*(2.*r/d1)**6 - 5.*(2.*r/d1)**4)*np.cos(4.*t)
            zernike += A6m2*np.sqrt(14.)*(15.*(2.*r/d1)**6 - 20.*(2.*r/d1)**4 - 6.*(2.*r/d1)**2)*np.sin(2.*t)
            zernike += A6p2*np.sqrt(14.)*(15.*(2.*r/d1)**6 - 20.*(2.*r/d1)**4 - 6.*(2.*r/d1)**2)*np.cos(2.*t)
            zernike += A60*np.sqrt(7.)*(20.*(2.*r/d1)**6 - 30.*(2.*r/d1)**4 +12*(2.*r/d1)**2 - 1)

            phase = 2.*np.pi/lam0*zernike*1.e-6

            # --- transform to image plane --- #
            complexPsf = np.fft.fftshift(np.fft.fft2(pupil * fiber
                                                     * np.exp(1j*phase)))
            return complexPsf[cc, cc]/np.abs(complexPsf[cc, cc]).max()

        zernikefile = resource_filename('gravipy', 'Phasemaps/' + zerfile)
        zer = np.load(zernikefile, allow_pickle=True).item()

        wave = self.wlSC

        if self.tel == 'UT':
            stopB = 8.0
            stopS = 0.96
            dalpha = 1
            totN = 1024
            d = 8
            amax = 100
            set_smooth = smooth

        elif self.tel == 'AT':
            stopB = 8.0/4.4
            stopS = 8.0/4.4*0.076
            dalpha = 1.*4.4
            totN = 1024
            d = 1.8
            amax = 100*4.4
            set_smooth = smooth  # / 4.4

        kernel = Gaussian2DKernel(x_stddev=smooth)

        print('Creating phasemaps:')
        print('StopB : %.2f' % stopB)
        print('StopS : %.2f' % stopS)
        print('Smooth: %.2f' % set_smooth)
        print('amax: %i' % amax)

        if nthreads == 1:
            all_pm = np.zeros((len(wave), 4, 201, 201),
                            dtype=np.complex_)
            all_pm_denom = np.zeros((len(wave), 4, 201, 201),
                                    dtype=np.complex_)
            for wdx, wl in enumerate(wave):
                print_status(wdx, len(wave))
                for GV in range(4):
                    zer_GV = zer['GV%i' % (GV+1)]
                    pm = phase_screen(*zer_GV, lam0=wl, d1=d, stopB=stopB,
                                      stopS=stopS, dalpha=dalpha, totN=totN,
                                      amax=amax)
                    if pm.shape != (201, 201):
                        print(pm.shape)
                        print('Need to convert to (201,201) shape')
                        pm = procrustes(pm, (201, 201), padval=0)
                    pm_sm = signal.convolve2d(pm, kernel, mode='same')
                    pm_sm_denom = signal.convolve2d(np.abs(pm)**2, 
                                                    kernel, mode='same')

                    all_pm[wdx, GV] = pm_sm
                    all_pm_denom[wdx, GV] = pm_sm_denom

                    if plot and wdx == 0:
                        plt.imshow(np.abs(pm_sm))
                        plt.colorbar()
                        plt.show()
                        plt.imshow(np.angle(pm_sm))
                        plt.colorbar()
                        plt.show()

        else:
            def multi_pm(lam):
                print(lam)
                m_all_pm = np.zeros((4, 201, 201), dtype=np.complex_)
                m_all_pm_denom = np.zeros((4, 201, 201), dtype=np.complex_)
                for GV in range(4):
                    zer_GV = zer['GV%i' % (GV+1)]
                    pm = phase_screen(*zer_GV, lam0=lam, d1=d, stopB=stopB,
                                      stopS=stopS, dalpha=dalpha, totN=totN,
                                      amax=amax)

                    if pm.shape != (201, 201):
                        print('Need to convert to (201,201) shape')
                        print(pm.shape)
                        pm = procrustes(pm, (201,201), padval=0)

                    pm_sm = signal.convolve2d(pm, kernel, mode='same')
                    pm_sm_denom = signal.convolve2d(np.abs(pm)**2, 
                                                    kernel, mode='same')
                    m_all_pm[GV] = pm_sm
                    m_all_pm_denom[GV] = pm_sm_denom
                return np.array([m_all_pm, m_all_pm_denom])

            res = np.array(Parallel(n_jobs=nthreads)(delayed(multi_pm)(lam) for lam in wave))

            all_pm = res[:, 0, :, :, :]
            all_pm_denom = res[:, 1, :, :, :]
        if datayear == 2019:
            savename = ('Phasemaps/Phasemap_%s_%s_Smooth%i.npy'
                        % (self.tel, self.resolution, smooth))
            savename2 = ('Phasemaps/Phasemap_%s_%s_Smooth%i_denom.npy'
                         % (self.tel, self.resolution, smooth))
        else:
            savename = ('Phasemaps/Phasemap_%s_%s_Smooth%i_2020data.npy'
                        % (self.tel, self.resolution, smooth))
            savename2 = ('Phasemaps/Phasemap_%s_%s_Smooth%i_2020data_denom.npy'
                         % (self.tel, self.resolution, smooth))
        savefile = resource_filename('gravipy', savename)
        np.save(savefile, all_pm)
        savefile = resource_filename('gravipy', savename2)
        np.save(savefile, all_pm_denom)

    def rotation(self, ang):
        """
        Rotation matrix, needed for phasemaps
        """
        return np.array([[np.cos(ang), np.sin(ang)],
                         [-np.sin(ang), np.cos(ang)]])

    def loadPhasemaps(self, interp, tofits=False):
        smoothkernel = self.smoothkernel
        datayear = self.datayear
        if datayear == 2019:
            pm1_file = ('Phasemaps/Phasemap_%s_%s_Smooth%i.npy'
                        % (self.tel, self.resolution, smoothkernel))
            pm2_file = ('Phasemaps/Phasemap_%s_%s_Smooth%i_denom.npy'
                        % (self.tel, self.resolution, smoothkernel))
        elif datayear == 2020:
            pm1_file = ('Phasemaps/Phasemap_%s_%s_Smooth%i_2020data.npy'
                        % (self.tel, self.resolution, smoothkernel))
            pm2_file = ('Phasemaps/Phasemap_%s_%s_Smooth%i_2020data_denom.npy'
                        % (self.tel, self.resolution, smoothkernel))

        try:
            pm1 = np.load(resource_filename('gravipy', pm1_file))
            pm2 = np.real(np.load(resource_filename('gravipy', pm2_file)))
        except FileNotFoundError:
            raise ValueError('%s does not exist, you have to create'
                             'the phasemap first!' % pm1_file)

        wave = self.wlSC
        if pm1.shape[0] != len(wave):
            print(pm1_file)
            print(pm1.shape[0], len(wave))
            raise ValueError('Phasemap and data have different num of channels')

        amp_map = np.abs(pm1)
        pha_map = np.angle(pm1, deg=True)
        amp_map_denom = pm2

        for wdx in range(len(wave)):
            for tel in range(4):
                amp_map[wdx, tel] /= np.max(amp_map[wdx, tel])
                amp_map_denom[wdx, tel] /= np.max(amp_map_denom[wdx, tel])

        if tofits:
            primary_hdu = fits.PrimaryHDU()
            hlist = [primary_hdu]
            for tel in range(4):
                hlist.append(fits.ImageHDU(amp_map[:, tel],
                                           name='SC_AMP UT%i' % (4-tel)))
                hlist.append(fits.ImageHDU(pha_map[:, tel],
                                           name='SC_PHA UT%i' % (4-tel)))
            hdul = fits.HDUList(hlist)
            hdul.writeto(resource_filename('gravipy', 'testfits.fits'),
                         overwrite=True)
            print('Saving phasemaps as fits file to: %s'
                  % resource_filename('gravipy', 'testfits.fits'))

        if interp:
            x = np.arange(201)
            y = np.arange(201)
            itel = np.arange(4)
            iwave = np.arange(len(wave))
            points = (iwave, itel, x, y)

            self.amp_map_int = interpolate.RegularGridInterpolator(points, amp_map)
            self.pha_map_int = interpolate.RegularGridInterpolator(points, pha_map)
            self.amp_map_denom_int = interpolate.RegularGridInterpolator(points, amp_map_denom)

        else:
            self.amp_map = amp_map
            self.pha_map = pha_map
            self.amp_map_denom = amp_map_denom

    def readPhasemaps(self, ra, dec, fromFits=True,
                      northangle=None, dra=None, ddec=None,
                      interp=True, givepos=False):
        """
        Calculates coupling amplitude / phase for given coordinates
        ra,dec: RA, DEC position on sky relative to nominal
                field center = SOBJ [mas]
        dra,ddec: ESO QC MET SOBJ DRA / DDEC:
            location of science object (= desired science fiber position,
                                        = field center)
            given by INS.SOBJ relative to *actual* fiber position measured
            by the laser metrology [mas]
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

        pm_pos = np.zeros((4, 2))
        readout_pos = np.zeros((4*len(wave),4))
        readout_pos[:, 0] = np.tile(np.arange(len(wave)), 4)
        readout_pos[:, 1] = np.repeat(np.arange(4), len(wave))

        for tel in range(4):
            pos = np.array([ra + dra[tel], dec + ddec[tel]])
            if self.tel == 'AT':
                pos /= 4.4
            try:
                pos[0] += self.pm_pos_off[0]
                pos[1] += self.pm_pos_off[1]
            except (NameError, AttributeError):
                pass
            pos_rot = np.dot(self.rotation(northangle[tel]), pos) + 100
            readout_pos[readout_pos[:, 1] == tel, 2] = pos_rot[1]
            readout_pos[readout_pos[:, 1] == tel, 3] = pos_rot[0]
            pm_pos[tel] = pos_rot

        cor_amp = self.amp_map_int(readout_pos).reshape(4, len(wave))
        cor_pha = self.pha_map_int(readout_pos).reshape(4, len(wave))
        cor_int_denom = self.amp_map_denom_int(readout_pos).reshape(4, len(wave))

        if givepos:
            return readout_pos
        else:
            return cor_amp, cor_pha, cor_int_denom

    def phasemap_source(self, x, y, northA, dra, ddec):
        amp, pha, inten = self.readPhasemaps(x, y, fromFits=False,
                                             northangle=northA, dra=dra,
                                             ddec=ddec, interp=self.interppm)
        pm_amp = np.array([[amp[0], amp[1]],
                           [amp[0], amp[2]],
                           [amp[0], amp[3]],
                           [amp[1], amp[2]],
                           [amp[1], amp[3]],
                           [amp[2], amp[3]]])
        pm_pha = np.array([[pha[0], pha[1]],
                           [pha[0], pha[2]],
                           [pha[0], pha[3]],
                           [pha[1], pha[2]],
                           [pha[1], pha[3]],
                           [pha[2], pha[3]]])
        pm_int = np.array([[inten[0], inten[1]],
                           [inten[0], inten[2]],
                           [inten[0], inten[3]],
                           [inten[1], inten[2]],
                           [inten[1], inten[3]],
                           [inten[2], inten[3]]])
        return pm_amp, pm_pha, pm_int


@jit(nopython=True)
def _rotation(ang):
    """
    Rotation matrix, needed for phasemaps
    """
    return np.array([[np.cos(ang), np.sin(ang)],
                     [-np.sin(ang), np.cos(ang)]])


def _readPhasemaps(ra, dec, northangle, amp_map_int, pha_map_int,
                   amp_map_denom_int, wave,
                   dra=np.zeros(4), ddec=np.zeros(4)):
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
    pm_pos = np.zeros((4, 2))
    readout_pos = np.zeros((4*len(wave), 4))
    readout_pos[:, 0] = np.tile(np.arange(len(wave)), 4)
    readout_pos[:, 1] = np.repeat(np.arange(4), len(wave))

    for tel in range(4):
        pos = np.array([ra + dra[tel], dec + ddec[tel]])
        try:
            pos[0] += self.pm_pos_off[0]
            pos[1] += self.pm_pos_off[1]
        except (NameError, AttributeError):
            pass
        pos_rot = np.dot(_rotation(northangle[tel]), pos) + 100
        readout_pos[readout_pos[:, 1] == tel, 2] = pos_rot[1]
        readout_pos[readout_pos[:, 1] == tel, 3] = pos_rot[0]
        pm_pos[tel] = pos_rot

    amp = amp_map_int(readout_pos).reshape(4, len(wave))
    pha = pha_map_int(readout_pos).reshape(4, len(wave))
    inten = amp_map_denom_int(readout_pos).reshape(4, len(wave))

    cor_amp = np.array([[amp[0], amp[1]],
                        [amp[0], amp[2]],
                        [amp[0], amp[3]],
                        [amp[1], amp[2]],
                        [amp[1], amp[3]],
                        [amp[2], amp[3]]])
    cor_pha = np.array([[pha[0], pha[1]],
                        [pha[0], pha[2]],
                        [pha[0], pha[3]],
                        [pha[1], pha[2]],
                        [pha[1], pha[3]],
                        [pha[2], pha[3]]])
    cor_int_denom = np.array([[inten[0], inten[1]],
                              [inten[0], inten[2]],
                              [inten[0], inten[3]],
                              [inten[1], inten[2]],
                              [inten[1], inten[3]],
                              [inten[2], inten[3]]])
    return cor_amp, cor_pha, cor_int_denom


@jit(nopython=True)
def _vis_intensity_approx(s, alpha, lambda0, dlambda):
    """
    Approximation for Modulated interferometric intensity
    s:      B*skypos-opd1-opd2
    alpha:  power law index
    lambda0:zentral wavelength
    dlambda:size of channels
    """
    x = 2*s*dlambda/lambda0**2.
    sinc = np.sinc(x)  # be aware that np.sinc = np.sin(pi*x)/(pi*x)
    return (lambda0/2.2)**(-1-alpha)*2*dlambda*sinc*np.exp(-2.j*np.pi*s/lambda0)


#@jit(nopython=False)
def _vis_intensity(s, alpha, lambda0, dlambda):
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
                    res[idx] = _vis_intensity_num(s[idx], alpha,
                                                        lambda0[idx], dlambda[idx])
                else:
                    up = _vis_int_full(s[idx], alpha, x1[idx])
                    low = _vis_int_full(s[idx], alpha, x2[idx])
                    res[idx] = up - low
        else:
            res = np.zeros(len(lambda0), dtype=np.complex_)
            for idx in range(len(lambda0)):
                if s == 0 and alpha == 0:
                    res[idx] = _vis_intensity_num(s, alpha, lambda0[idx],
                                                        dlambda[idx])
                else:
                    up = _vis_int_full(s, alpha, x1[idx])
                    low = _vis_int_full(s, alpha, x2[idx])
                    res[idx] = up - low
    else:
        if s == 0 and alpha == 0:
            res = _vis_intensity_num(s, alpha, lambda0, dlambda)
        else:
            up = _vis_int_full(s, alpha, x1)
            low = _vis_int_full(s, alpha, x2)
            res = up - low
    return res


#@jit(nopython=True)
def _vis_int_full(s, alpha, difflam):
    if s == 0:
        return -2.2**(1 + alpha)/alpha*difflam**(-alpha)
    a = difflam*(difflam/2.2)**(-1-alpha)
    bval = mpmath.gammainc(alpha, (2*1j*np.pi*s/difflam))
    b = float(bval.real)+float(bval.imag)*1j
    c = (2*np.pi*1j*s/difflam)**alpha
    return (a*b/c)


#@jit(nopython=True)
def _visibility_integrator(wave, s, alpha):
    """
    complex integral to be integrated over wavelength
    wave in [micron]
    theta holds the exponent alpha, and the seperation s
    """
    return (wave/2.2)**(-1-alpha)*np.exp(-2*np.pi*1j*s/wave)


#@jit(nopython=True)
def _vis_intensity_num(s, alpha, lambda0, dlambda):
    """
    Dull numeric solution for Modulated interferometric intensity
    s:      B*skypos-opd1-opd2
    alpha:  power law index
    lambda0:zentral wavelength
    dlambda:size of channels
    """
    if np.all(s == 0.) and alpha != 0:
        return np.complex128(-2.2**(1 + alpha)/alpha*(lambda0+dlambda)**(-alpha) - (-2.2**(1 + alpha)/alpha*(lambda0-dlambda)**(-alpha)))
    else:
        return complex_quadrature_num(_visibility_integrator, lambda0-dlambda, lambda0+dlambda, (s, alpha))


def _ind_visibility(s, alpha, wave, dlambda, fit_mode):
    mode = fit_mode
    if mode == "approx":
        ind_vis = _vis_intensity_approx(s, alpha, wave, dlambda)
    elif mode == "analytic":
        ind_vis = _vis_intensity(s, alpha, wave, dlambda)
    elif mode == "numeric":
        ind_vis = _vis_intensity_num(s, alpha, wave, dlambda)
    else:
        raise ValueError('approx has to be approx, analytic or numeric')
    return ind_vis


def _lnprob_mstars(theta, fitdata, lower, upper, fitarg, fithelp):
    if np.any(theta < lower) or np.any(theta > upper):
        return -np.inf
    return _lnlike_mstars(theta, fitdata, fitarg, fithelp)


def _lnlike_mstars(theta, fitdata, fitarg, fithelp):

    (nsource, fit_for, bispec_ind, fit_mode, wave, dlambda,
     fixedBHalpha, todel, fixed, phasemaps, northA, dra, ddec, amp_map_int,
     pha_map_int, amp_map_denom_int, fit_phasemaps, fix_pm_sources,
     fix_pm_amp_c, fix_pm_pha_c, fix_pm_int_c) = fithelp

    for ddx in range(len(todel)):
        theta = np.insert(theta, todel[ddx], fixed[ddx])

    model_visamp, model_visphi, model_closure = _calc_vis_mstars(theta, fitarg,
                                                                 fithelp)
    model_vis2 = model_visamp**2.

    (visamp, visamp_error, visamp_flag,
        vis2, vis2_error, vis2_flag,
        closure, closure_error, closure_flag,
        visphi, visphi_error, visphi_flag) = fitdata

    res_visamp = np.sum(-(model_visamp-visamp)**2/visamp_error**2*(1-visamp_flag))
    res_vis2 = np.sum(-(model_vis2-vis2)**2./vis2_error**2.*(1-vis2_flag))

    res_closure = np.degrees(np.abs(np.exp(1j*np.radians(model_closure))
                                    - np.exp(1j*np.radians(closure))))
    res_clos = np.sum(-res_closure**2./closure_error**2.*(1-closure_flag))

    res_visphi = np.degrees(np.abs(np.exp(1j*np.radians(model_visphi))
                                   - np.exp(1j*np.radians(visphi))))
    res_phi = np.sum(-res_visphi**2./visphi_error**2.*(1-visphi_flag))

    ln_prob_res = 0.5 * (res_visamp * fit_for[0]
                         + res_vis2 * fit_for[1]
                         + res_clos * fit_for[2]
                         + res_phi * fit_for[3])
    return ln_prob_res


def _calc_vis_mstars(theta_in, fitarg, fithelp):
    mas2rad = 1e-3 / 3600 / 180 * np.pi

    (nsource, fit_for, bispec_ind, fit_mode, wave, dlambda,
     fixedBHalpha, todel, fixed, phasemaps, northA, dra, ddec, amp_map_int,
     pha_map_int, amp_map_denom_int, fit_phasemaps, fix_pm_sources,
     fix_pm_amp_c, fix_pm_pha_c, fix_pm_int_c) = fithelp

    u = fitarg[0]
    v = fitarg[1]

    theta = theta_in
    # theta = np.zeros(nsource*3+10)
    # _th_count = 0
    # _fi_count = 0
    # for ndx in range(len(theta)):
    #     if ndx in todel:
    #         theta[ndx] = fixed[_fi_count]
    #         _fi_count += 1
    #     else:
    #         theta[ndx] = theta_in[_th_count]
    #         _th_count += 1

    th_rest = nsource*3-1
    alpha_SgrA = theta[th_rest]
    fluxRatioBG = theta[th_rest+1]
    pc_RA = theta[th_rest+2]
    pc_DEC = theta[th_rest+3]
    fr_BH = 10**(theta[th_rest+4])

    alpha_bg = 3.
    alpha_stars = 3

    if phasemaps:
        if fit_phasemaps:
            pm_sources = []
            pm_amp_c, pm_pha_c, pm_int_c = _readPhasemaps(pc_RA, pc_DEC, northA,
                                                          amp_map_int, pha_map_int,
                                                          amp_map_denom_int,
                                                          wave, dra, ddec)

            for ndx in range(nsource):
                if ndx == 0:
                    pm_amp, pm_pha, pm_int = _readPhasemaps(pc_RA + theta[0],
                                                            pc_DEC + theta[1],
                                                            northA, amp_map_int,
                                                            pha_map_int,
                                                            amp_map_denom_int,
                                                            wave, dra, ddec)
                    pm_sources.append([pm_amp, pm_pha, pm_int])
                else:
                    pm_amp, pm_pha, pm_int = _readPhasemaps(pc_RA + theta[ndx*3-1],
                                                            pc_DEC + theta[ndx*3],
                                                            northA, amp_map_int,
                                                            pha_map_int,
                                                            amp_map_denom_int,
                                                            wave, dra, ddec)
                    pm_sources.append([pm_amp, pm_pha, pm_int])
        else:
            pm_sources = fix_pm_sources
            pm_amp_c, pm_pha_c, pm_int_c = fix_pm_amp_c, fix_pm_pha_c, fix_pm_int_c

    vis = np.zeros((6, len(wave))) + 0j
    for i in range(0, 6):
        s_SgrA = ((pc_RA)*u[i] + (pc_DEC)*v[i]) * mas2rad * 1e6

        if phasemaps:
            s_SgrA -= ((pm_pha_c[i, 0] - pm_pha_c[i, 1])/360*wave)

        s_stars = []
        for ndx in range(nsource):
            if ndx == 0:
                s_s = ((theta[0] + pc_RA)*u[i]
                       + (theta[1] + pc_DEC)*v[i]) * mas2rad * 1e6
            else:
                s_s = ((theta[ndx*3-1] + pc_RA)*u[i]
                       + (theta[ndx*3] + pc_DEC)*v[i]) * mas2rad * 1e6

            if phasemaps:
                _, pm_pha, _ = pm_sources[ndx]
                s_s -= ((pm_pha[i, 0] - pm_pha[i, 1])/360*wave)
            s_stars.append(s_s)

        intSgrA = _ind_visibility(s_SgrA, alpha_SgrA, wave,
                                  dlambda[i, :], fit_mode)
        intSgrA_center = _ind_visibility(0, alpha_SgrA, wave,
                                         dlambda[i, :], fit_mode)

        nom = intSgrA * fr_BH

        denom1 = np.copy(intSgrA_center) * fr_BH
        denom2 = np.copy(intSgrA_center) * fr_BH

        int_star_center = _ind_visibility(0, alpha_stars, wave,
                                          dlambda[i, :], fit_mode)
        if phasemaps:
            pm_amp_norm, _, pm_int_norm = pm_sources[0]

            cr1 = (pm_amp_c[i, 0] / pm_amp_norm[i, 0])**2
            cr2 = (pm_amp_c[i, 1] / pm_amp_norm[i, 1])**2
            cr_denom1 = (pm_int_c[i, 0] / pm_int_norm[i, 0])
            cr_denom2 = (pm_int_c[i, 1] / pm_int_norm[i, 1])

            nom *= np.sqrt(cr1*cr2)
            denom1 *= cr_denom1
            denom2 *= cr_denom2

            for ndx in range(nsource):
                int_star = _ind_visibility(s_stars[ndx], alpha_stars, wave,
                                           dlambda[i, :], fit_mode)

                pm_amp, _, pm_int = pm_sources[ndx]
                cr1 = (pm_amp[i, 0] / pm_amp_norm[i, 0])**2
                cr2 = (pm_amp[i, 1] / pm_amp_norm[i, 1])**2
                cr_denom1 = (pm_int[i, 0] / pm_int_norm[i, 0])
                cr_denom2 = (pm_int[i, 1] / pm_int_norm[i, 1])

                if ndx == 0:
                    nom += (int_star)
                    denom1 += (int_star_center)
                    denom2 += (int_star_center)
                else:
                    nom += (10.**(theta[ndx*3+1]) * np.sqrt(cr1*cr2)
                            * int_star)
                    denom1 += (10.**(theta[ndx*3+1]) * cr_denom1
                               * int_star_center)
                    denom2 += (10.**(theta[ndx*3+1]) * cr_denom2
                               * int_star_center)

        else:
            for ndx in range(nsource):
                int_star = _ind_visibility(s_stars[ndx], alpha_stars, wave,
                                           dlambda[i, :], fit_mode)
                if ndx == 0:
                    nom += (int_star)
                    denom1 += (int_star_center)
                    denom2 += (int_star_center)
                else:
                    nom += (10.**(theta[ndx*3+2]) * int_star)
                    denom1 += (10.**(theta[ndx*3+2]) * int_star_center)
                    denom2 += (10.**(theta[ndx*3+2]) * int_star_center)

        intBG = _ind_visibility(0, alpha_bg, wave, dlambda[i, :], fit_mode)
        denom1 += (fluxRatioBG * intBG)
        denom2 += (fluxRatioBG * intBG)

        vis[i, :] = nom / (np.sqrt(denom1)*np.sqrt(denom2))
        vis[i, :] *= theta[th_rest+5+i]

    visamp = np.abs(vis)
    visphi = np.angle(vis, deg=True)
    closure = np.zeros((4, len(wave)))
    for idx in range(4):
        closure[idx] = (visphi[bispec_ind[idx,0]]
                        + visphi[bispec_ind[idx,1]]
                        - visphi[bispec_ind[idx,2]])

    visphi = visphi + 360.*(visphi < -180.) - 360.*(visphi>180.)
    closure = closure + 360.*(closure < -180.) - 360.*(closure>180.)
    return visamp, visphi, closure


class GravMFit(GravData, GravPhaseMaps):
    def __init__(self, data, verbose=False, ignore_tel=[]):
        super().__init__(data, verbose=verbose)
        self.getIntdata(ignore_tel=ignore_tel)

    def fitStars(self,
                 ra_list,
                 de_list,
                 fr_list,
                 fit_size=None,
                 fit_pos=None,
                 fit_fr=None,
                 nthreads=1,
                 nwalkers=301,
                 nruns=301,
                 bestchi=True,
                 bequiet=False,
                 fit_for=np.array([0.5, 0.5, 1.0, 0.0]),
                 fit_mode='numeric',
                 no_fit=False,
                 onlypol=None,
                 initial=None,
                 flagtill=3,
                 flagfrom=13,
                 fixedBHalpha=False,
                 fixedBG=False,
                 coh_loss=False,
                 plotCorner=True,
                 plotScience=True,
                 redchi2=False,
                 phasemaps=False,
                 fit_phasemaps=False,
                 interppm=True,
                 smoothkernel=15,
                 pmdatayear=2019):
        '''
        Multi source fit to GRAVITY data
        Function fits a central source and a number of companion sources.
        All flux ratios are with respect to centra source

        The length of the input lists defines number of companions!
        Flux input lists is number of companions - 1
        (Flux of first companion is set to 1)

        Mandatory argumens:
        ra_list:        Initial guess for ra separation of companions
        de_list:        Initial guess for dec separation of companions
        fr_list:        Initial guess for flux ratio of companions

        Optional arguments for companions:
        If those vaues are given they need to be a list with one entry per
        companion:
        fit_size:       Size of fitting area [5]
        fit_pos:        Fit position of each companion [True]
        fit_fr:         Fit flux ratio of each companion [True]

        Other optional arguments:
        nthreads:       number of cores [4]
        nwalkers:       number of walkers [500]
        nruns:          number of MCMC runs [500]
        bestchi:        Gives best chi2 (for True) or mcmc res as output [True]
        bequiet:        Suppresses ALL outputs
        fit_for:        weight of VA, V2, T3, VP [[0.5,0.5,1.0,0.0]]
        fit_mode:       Kind of integration for visibilities (approx, numeric,
                        analytic) [numeric]
        coh_loss:       If not None, fit for a coherence loss per Basline
                        Value is initial guess (0-1) [None]
        no_fit  :       Only gives fitting results for parameters from 
                        initial guess [False]
        onlypol:        Only fits one polarization for split mode, 
                        either 0 or 1 [None]
        initial:        Initial guess for fit [None]
        flagtill:       Flag blue channels, has to be changed for not LOW [3]
        flagfrom:       Flag red channels, has to be changed for not LOW [13]
        fixedBHalpha:   Fit for black hole power law [False]
        plotCorner:     plot MCMC results [True]
        plotScience:    plot fit result [True]
        redchi2:        Gives redchi2 instead of chi2 [False]
        phasemaps:      Use Phasemaps for fit [False]
        fit_phasemaps:  Fit phasemaps at each step, otherwise jsut takes the 
                        initial guess value [False]
        interppm:       Interpolate Phasemaps [True]
        smoothkernel:   Size of smoothing kernel in mas [15]
        simulate_pm:    Phasemaps for simulated data, 
                        sets ACQ parameter to 0 [False]
        '''

        if self.resolution != 'LOW' and flagtill == 3 and flagfrom == 13:
            raise ValueError('Initial values for flagtill and flagfrom have'
                             'to be changed if not low resolution')

        self.fit_for = fit_for
        self.fixedBHalpha = fixedBHalpha
        self.fixedBG = fixedBG
        self.coh_loss = coh_loss
        self.interppm = interppm
        self.fit_mode = fit_mode
        self.bequiet = bequiet
        self.phasemaps = phasemaps
        self.fit_phasemaps = fit_phasemaps
        self.datayear = pmdatayear
        self.smoothkernel = smoothkernel
        if phasemaps:
            self.loadPhasemaps(interp=interppm)

            header = fits.open(self.name)[0].header
            northangle1 = header['ESO QC ACQ FIELD1 NORTH_ANGLE']/180*math.pi
            northangle2 = header['ESO QC ACQ FIELD2 NORTH_ANGLE']/180*math.pi
            northangle3 = header['ESO QC ACQ FIELD3 NORTH_ANGLE']/180*math.pi
            northangle4 = header['ESO QC ACQ FIELD4 NORTH_ANGLE']/180*math.pi
            self.northangle = [northangle1, northangle2, 
                               northangle3, northangle4]

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

            if fit_phasemaps:
                phasemaps = GravPhaseMaps()
                phasemaps.tel = self.tel
                phasemaps.resolution = self.resolution
                phasemaps.smoothkernel = self.smoothkernel
                phasemaps.datayear = self.datayear
                phasemaps.wlSC = self.wlSC
                phasemaps.interppm = interppm
                phasemaps.loadPhasemaps(interp=interppm)

        nsource = len(ra_list)
        if fit_size is None:
            fit_size = np.ones(nsource)*5
        if fit_pos is None:
            fit_pos = np.ones(nsource)
        if fit_fr is None:
            fit_fr = np.ones(nsource-1)

        if len(de_list) != nsource or len(fit_pos) != nsource or len(fit_size) != nsource:
            raise ValueError('list of input parameters have different lengths')
        if len(fr_list) != (nsource-1) or len(fit_fr) != (nsource-1):
            raise ValueError('list of input parameters have different lengths,'
                             'fr list should be nsource-1')
        self.nsource = nsource

        # Get data from file
        tel = fits.open(self.name)[0].header["TELESCOP"]
        if tel == 'ESO-VLTI-U1234':
            self.tel = 'UT'
        elif tel == 'ESO-VLTI-A1234':
            self.tel = 'AT'
        else:
            raise ValueError('Telescope not AT or UT, something wrong'
                             'with input data')

        MJD = fits.open(self.name)[0].header["MJD-OBS"]
        u = self.u
        v = self.v
        wave = self.wlSC
        self.wave = wave
        self.getDlambda()

        results = []
        # Initial guesses
        if initial is not None:
            if len(initial) != 6:
                raise ValueError('Length of initial parameter list is'
                                 ' not correct')
            (alpha_SgrA_in, flux_ratio_bg_in, pc_RA_in, pc_DEC_in,
             flux_ratio_bh, coh_loss_in) = initial
        else:
            alpha_SgrA_in = -0.5
            flux_ratio_bg_in = 0.1
            pc_RA_in = 0
            pc_DEC_in = 0
            flux_ratio_bh = 1
            coh_loss_in = 1

        theta = np.zeros(nsource*3+10)
        lower = np.zeros(nsource*3+10)
        upper = np.zeros(nsource*3+10)

        theta_names = []
        todel = []
        fr_list = [np.log10(i) for i in fr_list]
        for ndx in range(nsource):
            if ndx == 0:
                # first star (no flux ratio)
                theta[0] = ra_list[0]
                theta[1] = de_list[0]
                lower[0] = ra_list[0] - fit_size[0]
                lower[1] = de_list[0] - fit_size[0]
                upper[0] = ra_list[0] + fit_size[0]
                upper[1] = de_list[0] + fit_size[0]
                if not fit_pos[0]:
                    todel.append(0)
                    todel.append(1)
                theta_names.append('dRA1')
                theta_names.append('dDEC1')
            else:
                theta[ndx*3-1] = ra_list[ndx]
                theta[ndx*3] = de_list[ndx]
                theta[ndx*3+1] = fr_list[ndx-1]

                lower[ndx*3-1] = ra_list[ndx] - fit_size[ndx]
                lower[ndx*3] = de_list[ndx] - fit_size[ndx]
                lower[ndx*3+1] = np.log10(0.001)

                upper[ndx*3-1] = ra_list[ndx] + fit_size[ndx]
                upper[ndx*3] = de_list[ndx] + fit_size[ndx]
                upper[ndx*3+1] = np.log10(100.)

                if not fit_pos[ndx]:
                    todel.append(ndx*3-1)
                    todel.append(ndx*3)
                if not fit_fr[ndx-1]:
                    todel.append(ndx*3+1)
                theta_names.append('dRA%i' % (ndx + 1))
                theta_names.append('dDEC%i' % (ndx + 1))
                theta_names.append('fr%i' % (ndx + 1))

        th_rest = nsource*3-1

        theta[th_rest] = alpha_SgrA_in
        theta[th_rest+1] = flux_ratio_bg_in
        theta[th_rest+2] = pc_RA_in
        theta[th_rest+3] = pc_DEC_in
        theta[th_rest+4] = np.log10(flux_ratio_bh)

        pc_size = 5
        lower[th_rest] = -10
        lower[th_rest+1] = 0.1
        lower[th_rest+2] = pc_RA_in - pc_size
        lower[th_rest+3] = pc_DEC_in - pc_size
        lower[th_rest+4] = np.log10(0.001)

        upper[th_rest] = 10
        upper[th_rest+1] = 20
        upper[th_rest+2] = pc_RA_in + pc_size
        upper[th_rest+3] = pc_DEC_in + pc_size
        upper[th_rest+4] = np.log10(100.)

        theta_names.append('alpha BH')
        theta_names.append('f BG')
        theta_names.append('pc RA')
        theta_names.append('pc Dec')
        theta_names.append('fr BH')

        theta[th_rest+5:] = coh_loss_in
        upper[th_rest+5:] = 1
        lower[th_rest+5:] = 0.1

        theta_names.append('CL1')
        theta_names.append('CL2')
        theta_names.append('CL3')
        theta_names.append('CL4')
        theta_names.append('CL5')
        theta_names.append('CL6')

        self.theta_names = theta_names

        ndim = len(theta)
        if fixedBHalpha:
            todel.append(th_rest)
        if fixedBG or coh_loss:
            todel.append(th_rest+1)
        if fit_for[3] == 0:
            todel.append(th_rest+2)
            todel.append(th_rest+3)
        if not coh_loss:
            todel.extend(np.arange(th_rest+5, th_rest+5+6))
        ndof = ndim - len(todel)
        self.theta_in = np.copy(theta)
        self.theta_allnames = np.copy(theta_names)
        self.theta_names = theta_names

        if phasemaps:
            if not self.fit_phasemaps:
                self.pm_sources = []
                self.pm_amp_c, self.pm_pha_c, self.pm_int_c = self.phasemap_source(0, 0,
                                                                        self.northangle, self.dra, self.ddec)

                pm_amp, pm_pha, pm_int = self.phasemap_source(0 + theta[0],
                                                            0 + theta[1],
                                                            self.northangle, self.dra, self.ddec)
                self.pm_sources.append([pm_amp, pm_pha, pm_int])
                for ndx in range(1, nsource):
                    pm_amp, pm_pha, pm_int = self.phasemap_source(0 + theta[ndx*3-1],
                                                                0 + theta[ndx*3],
                                                                self.northangle, self.dra, self.ddec)
                    self.pm_sources.append([pm_amp, pm_pha, pm_int])

        if no_fit:
            plotCorner = False
            print('Will not fit the data, just print out the results for the '
                  'given initial conditions')

        for ddx in sorted(todel, reverse=True):
            del theta_names[ddx]
        fixed = theta[todel]
        theta = np.delete(theta, todel)
        lower = np.delete(lower, todel)
        upper = np.delete(upper, todel)
        self.fixed = fixed
        self.todel = todel

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
            if onlypol is not None:
                polnom = [onlypol]
            else:
                polnom = [0, 1]

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
            polnom = [0]

        for dit in range(ndit):
            if not bequiet and not no_fit:
                print('Run MCMC for DIT %i' % (dit+1))
            ditstart = dit*6
            ditstop = ditstart + 6
            t3ditstart = dit*4
            t3ditstop = t3ditstart + 4

            for idx in polnom:
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
                closamp = np.nan_to_num(closamp)
                closamp_error[closamp_flag] = 1.

                with np.errstate(invalid='ignore'):
                    vis2_flag1 = (vis2 > 1) | (vis2 < 1.e-5)
                vis2_flag2 = np.isnan(vis2)
                vis2_flag_final = ((vis2_flag) | (vis2_flag1) | (vis2_flag2))
                vis2_flag = vis2_flag_final
                vis2 = np.nan_to_num(vis2)
                vis2_error[vis2_flag] = 1.

                closure = np.nan_to_num(closure)
                visphi = np.nan_to_num(visphi)
                visphi_flag[np.where(visphi_error == 0)] = True
                visphi_error[np.where(visphi_error == 0)] = 100
                closure_flag[np.where(closure_error == 0)] = True
                closure_error[np.where(closure_error == 0)] = 100

                if ((flagtill > 0) and (flagfrom > 0)):
                    p = flagtill
                    t = flagfrom
                    if idx == 0 and dit == 0:
                        if not bequiet:
                            print('using channels from #%i to #%i' % (p, t))
                    visamp_flag[:, 0:p] = True
                    vis2_flag[:, 0:p] = True
                    visphi_flag[:, 0:p] = True
                    closure_flag[:, 0:p] = True
                    closamp_flag[:, 0:p] = True

                    visamp_flag[:, t:] = True
                    vis2_flag[:, t:] = True
                    visphi_flag[:, t:] = True
                    closure_flag[:, t:] = True
                    closamp_flag[:, t:] = True

                width = 1e-1
                ndim = len(theta)
                pos = np.ones((nwalkers, ndim))
                for par in range(ndim):
                    pos[:, par] = (theta[par]
                                   + width*np.random.randn(nwalkers))

                if not bequiet:
                    if not no_fit:
                        print('Run MCMC for Pol %i' % (idx+1))
                    else:
                        print('Pol %i' % (idx+1))

                fitdata = [visamp, visamp_error, visamp_flag,
                           vis2, vis2_error, vis2_flag,
                           closure, closure_error, closure_flag,
                           visphi, visphi_error, visphi_flag]
                fitarg = [u, v]

                if self.phasemaps:
                    if fit_phasemaps:
                        fithelp = [self.nsource, self.fit_for, self.bispec_ind,
                                   self.fit_mode, self.wave, self.dlambda,
                                   self.fixedBHalpha,
                                   todel, fixed,
                                   self.phasemaps, self.northangle, self.dra,
                                   self.ddec, phasemaps.amp_map_int,
                                   phasemaps.pha_map_int, 
                                   phasemaps.amp_map_denom_int,
                                   fit_phasemaps, None, None, None, None]
                    else:
                        fithelp = [self.nsource, self.fit_for, self.bispec_ind,
                                   self.fit_mode, self.wave, self.dlambda,
                                   self.fixedBHalpha,
                                   todel, fixed,
                                   self.phasemaps, self.northangle, self.dra,
                                   self.ddec, None, None, None,
                                   fit_phasemaps, self.pm_sources, 
                                   self.pm_amp_c, self.pm_pha_c, self.pm_int_c]
                else:
                    fithelp = [self.nsource, self.fit_for, self.bispec_ind,
                               self.fit_mode, self.wave, self.dlambda,
                               self.fixedBHalpha,
                               todel, fixed,
                               self.phasemaps, None, None, None, None, None,
                               None, None, None, None, None, None]

                if not no_fit:
                    if nthreads == 1:
                        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                        _lnprob_mstars,
                                                        args=(fitdata, lower,
                                                              upper, fitarg,
                                                              fithelp))
                        if bequiet:
                            sampler.run_mcmc(pos, nruns, progress=False)
                        else:
                            sampler.run_mcmc(pos, nruns, progress=True)
                    else:
                        with Pool(processes=nthreads) as pool:
                            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                            _lnprob_mstars,
                                                            args=(fitdata,
                                                                  lower,
                                                                  upper, fitarg,
                                                                  fithelp),
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

                    samples = sampler.chain
                    mostprop = sampler.flatchain[np.argmax(sampler.flatlnprobability)]

                    clsamples = samples
                    cllabels = theta_names
                    clmostprop = mostprop

                    cldim = len(cllabels)
                    if plotCorner:
                        fig, axes = plt.subplots(cldim, figsize=(8, cldim/1.5),
                                                 sharex=True)
                        for i in range(cldim):
                            ax = axes[i]
                            ax.plot(clsamples[:, :, i].T, "k", alpha=0.3)
                            ax.set_ylabel(theta_names[i])
                            ax.yaxis.set_label_coords(-0.1, 0.5)
                        axes[-1].set_xlabel("step number")
                        plt.show()

                    if nruns > 300:
                        fl_samples = samples[:, -200:, :].reshape((-1, ndim))
                    elif nruns > 200:
                        fl_samples = samples[:, -100:, :].reshape((-1, ndim))
                    else:
                        fl_samples = samples.reshape((-1, ndim))

                    if plotCorner:
                        fig = corner.corner(fl_samples,
                                            quantiles=[0.16, 0.5, 0.84],
                                            truths=mostprop,
                                            labels=theta_names)
                        plt.show()

                    # get the actual fit
                    theta_fit = np.percentile(fl_samples, [50], axis=0).T.flatten()
                    percentiles = np.percentile(fl_samples, [16, 50, 84], axis=0).T
                    mostlike_m = percentiles[:,1] - percentiles[:,0]
                    mostlike_p = percentiles[:,2] - percentiles[:,1]
                    if bestchi:
                        theta_result = mostprop
                    else:
                        theta_result = theta_fit
                else:
                    theta_result = theta

                self.theta_result = theta_result

                results.append(theta_result)
                fulltheta = np.copy(theta_result)
                all_mostprop = np.copy(mostprop)
                all_mostlike = np.copy(theta_fit)
                for ddx in range(len(todel)):
                    fulltheta = np.insert(fulltheta, todel[ddx], fixed[ddx])
                    all_mostprop = np.insert(all_mostprop, todel[ddx],
                                             fixed[ddx])
                    all_mostlike = np.insert(all_mostlike, todel[ddx],
                                             fixed[ddx])
                    mostlike_m = np.insert(mostlike_m, todel[ddx], 0)
                    mostlike_p = np.insert(mostlike_p, todel[ddx], 0)

                if idx == 0 and dit == 0:
                    fittab = pd.DataFrame()
                _fittab = pd.DataFrame()
                _fittab["column"] = ["in P%i_%i" % (idx, dit),
                                     "M.L. P%i_%i" % (idx, dit),
                                     "M.P. P%i_%i" % (idx, dit),
                                     "$-\sigma$ P%i_%i" % (idx, dit),
                                     "$+\sigma$ P%i_%i" % (idx, dit)]
                for ndx, name in enumerate(self.theta_allnames):
                    _fittab[name] = pd.Series([self.theta_in[ndx],
                                               all_mostprop[ndx],
                                               all_mostlike[ndx],
                                               mostlike_m[ndx],
                                               mostlike_p[ndx]])
                fittab = fittab.append(_fittab, ignore_index=True)

                (fit_visamp, fit_visphi,
                 fit_closure) = _calc_vis_mstars(fulltheta, fitarg, fithelp)
                fit_vis2 = fit_visamp**2.

                self.result_fit_visamp = fit_visamp
                self.result_fit_vis2 = fit_vis2
                self.result_visphi = fit_visphi
                self.result_closure = fit_closure

                res_visamp = fit_visamp-visamp
                res_vis2 = fit_vis2-vis2

                res_closure = np.degrees(np.abs(np.exp(1j*np.radians(fit_closure)) - np.exp(1j*np.radians(closure))))
                res_closure = np.sum(-res_closure**2./closure_error**2.*(1-closure_flag))

                res_visphi = np.degrees(np.abs(np.exp(1j*np.radians(fit_visphi)) - np.exp(1j*np.radians(visphi))))
                res_visphi = np.sum(-res_visphi**2./visphi_error**2.*(1-visphi_flag))

                redchi_visamp = np.sum(res_visamp**2./visamp_error**2.*(1-visamp_flag))
                redchi_vis2 = np.sum(res_vis2**2./vis2_error**2.*(1-vis2_flag))
                redchi_closure = np.sum(res_closure**2./closure_error**2.*(1-closure_flag))
                redchi_visphi = np.sum(res_visphi**2./visphi_error**2.*(1-visphi_flag))

                if redchi2:
                    redchi_visamp /= (visamp.size-np.sum(visamp_flag)-ndof)
                    redchi_vis2 /= (vis2.size-np.sum(vis2_flag)-ndof)
                    redchi_closure /= (closure.size-np.sum(closure_flag)-ndof)
                    redchi_visphi /= (visphi.size-np.sum(visphi_flag)-ndof)
                    chi2string = 'red. chi2'
                else:
                    chi2string = 'chi2'

                redchi = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi]
                if idx == 0:
                    redchi0 = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi]
                    self.redchi0 = redchi0
                elif idx == 1:
                    redchi1 = [redchi_visamp, redchi_vis2, redchi_closure, redchi_visphi]
                    self.redchi1 = redchi1

                if not bequiet:
                    print('\n')
                    print('ndof: %i' % (vis2.size-np.sum(vis2_flag)-ndof))
                    print(chi2string + " for visamp: %.2f" % redchi_visamp)
                    print(chi2string + " for vis2: %.2f" % redchi_vis2)
                    print(chi2string + " for visphi: %.2f" % redchi_visphi)
                    print(chi2string + " for closure: %.2f" % redchi_closure)
                    print('\n')

                if not no_fit:
                    percentiles = np.percentile(fl_samples,
                                                [16, 50, 84], axis=0).T
                    percentiles[:, 0] = percentiles[:, 1] - percentiles[:, 0]
                    percentiles[:, 2] = percentiles[:, 2] - percentiles[:, 1]

                    if not bequiet:
                        print("-----------------------------------")
                        print("Best chi2 result:")
                        for i in range(0, cldim):
                            print("%s = %.3f" % (cllabels[i], clmostprop[i]))
                        print("\n")
                        print("MCMC Result:")
                        for i in range(0, cldim):
                            print("%s = %.3f + %.3f - %.3f" % (cllabels[i],
                                                            percentiles[i,1],
                                                            percentiles[i,2],
                                                            percentiles[i,0]))
                        print("-----------------------------------")

                if plotScience:
                    if idx == 0:
                        plotdata = []
                    plotdata.append([theta_result, fitdata, fitarg, fithelp])

            if plotScience:
                self.plotFitComb(plotdata)
        self.fittab = fittab

        try:
            fitted = 1-(np.array(self.fit_for) == 0)
            redchi0_f = np.sum(redchi0*fitted)
            if onlypol == 0:
                redchi1 = np.zeros_like(redchi0)
            redchi1_f = np.sum(redchi1*fitted)
            redchi_f = redchi0_f + redchi1_f
            if not bequiet:
                print('Combined %s of fitted data: %.3f' % (chi2string,
                                                            redchi_f))
        except UnboundLocalError:
            pass
        except:
            print("could not compute reduced chi2")
        if onlypol is not None and ndit == 1:
            return theta_result
        else:
            return results

    def plotFitComb(self, plotdata, nicer=True):
        rad2as = 180 / np.pi * 3600
        stname = self.name.find('GRAVI')
        title_name = self.name[stname:-5]

        nplot = len(plotdata)
        if nplot == 2:
            plotsplit = True
        else:
            plotsplit = False

        wave = self.wlSC
        dlambda = self.dlambda
        if self.phasemaps:
            wave_model = wave
        else:
            wave_model = np.linspace(wave[0], wave[len(wave)-1], 1000)
        dlambda_model = np.zeros((6, len(wave_model)))
        for i in range(0, 6):
            dlambda_model[i, :] = np.interp(wave_model, wave, dlambda[i, :])

        u = self.u
        v = self.v
        magu = np.sqrt(u**2.+v**2.)

        u_as_model = np.zeros((len(u),len(wave_model)))
        v_as_model = np.zeros((len(v),len(wave_model)))
        for i in range(0, len(u)):
            u_as_model[i, :] = u[i]/(wave_model*1.e-6) / rad2as
            v_as_model[i, :] = v[i]/(wave_model*1.e-6) / rad2as
        magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)

        fitres = []
        for idx in range(nplot):
            theta, fitdata, fitarg, fithelp = plotdata[idx]

            # (visamp, visamp_error, visamp_flag,
            #  vis2, vis2_error, vis2_flag,
            #  closure, closure_error, closure_flag,
            #  visphi, visphi_error, visphi_flag) = fitdata

            (nsource, fit_for, bispec_ind, fit_mode, wave, dlambda,
             fixedBHalpha, todel, fixed, phasemaps, northA, dra, ddec, amp_map_int,
             pha_map_int, amp_map_denom_int, fit_phasemaps, fix_pm_sources,
             fix_pm_amp_c, fix_pm_pha_c, fix_pm_int_c) = fithelp

            for ddx in range(len(todel)):
                theta = np.insert(theta, todel[ddx], fixed[ddx])

            fithelp[4] = wave_model
            fithelp[5] = dlambda_model
            self.wave = wave_model
            self.dlambda = dlambda_model
            fitres.append(_calc_vis_mstars(theta, fitarg, fithelp))

        self.wave = wave
        self.dlambda = dlambda
        magu_as = np.copy(self.spFrequAS)
        magu_as_T3 = np.copy(self.spFrequAS_T3)
        magu_as_T3_model = np.zeros((4, len(wave_model)))

        if nicer:
            bl_sort = [2, 3, 5, 0, 4, 1]
            cl_sort = [0, 3, 2, 1]
            nchannel = len(magu_as[0])
            for bl in range(6):
                magu_as[bl] = (np.linspace(nchannel, 0, nchannel)
                               + bl_sort[bl]*(nchannel+nchannel//2))
                magu_as_model[bl] = (np.linspace(nchannel, 0, len(wave_model))
                                     + bl_sort[bl]*(nchannel+nchannel//2))
            for cl in range(4):
                magu_as_T3[cl] = (np.linspace(nchannel, 0, nchannel)
                                  + cl_sort[cl]*(nchannel+nchannel//2))
                magu_as_T3_model[cl] = (np.linspace(nchannel, 0,
                                                    len(wave_model))
                                        + cl_sort[cl]*(nchannel+nchannel//2))
        else:
            for cl in range(4):
                magu_as_T3_model[cl] = (self.max_spf[cl]/(wave_model*1.e-6)
                                        / rad2as)

        # Visamp
        if self.fit_for[0]:
            plt.figure(figsize=(10, 5))
            if plotsplit:
                gs = gridspec.GridSpec(1,2, wspace=0.05)
            for idx in range(nplot):
                if plotsplit:
                    ax = plt.subplot(gs[0, idx])
                visamp = plotdata[idx][1][0]
                visamp_error = plotdata[idx][1][1]
                visamp_flag = plotdata[idx][1][2]
                model_visamp_full = fitres[idx][0]
                for i in range(0, 6):
                    plt.errorbar(magu_as[i, :],
                                 visamp[i, :]*(1-visamp_flag)[i],
                                 visamp_error[i, :]*(1-visamp_flag)[i],
                                 color=self.colors_baseline[i],
                                 ls='', lw=1, alpha=0.5, capsize=0)
                    plt.scatter(magu_as[i,:],
                                visamp[i, :]*(1-visamp_flag)[i],
                                color=self.colors_baseline[i],
                                alpha=0.5, label=self.baseline_labels[i])
                    if nicer:
                        plt.text(magu_as[i, :].mean(), -0.07,
                                 self.baseline_labels[i],
                                 color=self.colors_baseline[i],
                                 ha='center', va='center')
                    plt.plot(magu_as_model[i, :], model_visamp_full[i, :],
                             color='grey', zorder=100)
                if idx == 0:
                    plt.ylabel('Visibility Amplitude')
                else:
                    ax.set_yticklabels([])
                plt.ylim(-0.03, 1.1)
                if nicer:
                    # ax.set_xticklabels([])
                    ax.set_xticks([])
                else:
                    plt.legend()
                    plt.xlabel('spatial frequency (1/arcsec)')
            plt.suptitle(title_name, y=0.92)
            plt.show()

        # Vis2
        if self.fit_for[1]:
            plt.figure(figsize=(10, 5))
            if plotsplit:
                gs = gridspec.GridSpec(1, 2, wspace=0.05)
            for idx in range(nplot):
                if plotsplit:
                    ax = plt.subplot(gs[0, idx])
                vis2 = plotdata[idx][1][3]
                vis2_error = plotdata[idx][1][4]
                vis2_flag = plotdata[idx][1][5]
                model_vis2_full = fitres[idx][0]**2
                for i in range(0,6):
                    plt.errorbar(magu_as[i, :],
                                 vis2[i, :]*(1-vis2_flag)[i],
                                 vis2_error[i, :]*(1-vis2_flag)[i],
                                 color=self.colors_baseline[i],
                                 ls='', lw=1, alpha=0.5, capsize=0)
                    plt.scatter(magu_as[i,:],
                                vis2[i, :]*(1-vis2_flag)[i],
                                color=self.colors_baseline[i],
                                alpha=0.5, label=self.baseline_labels[i])
                    if nicer:
                        plt.text(magu_as[i, :].mean(), -0.07,
                                 self.baseline_labels[i],
                                 color=self.colors_baseline[i],
                                 ha='center', va='center')
                    plt.plot(magu_as_model[i, :], model_vis2_full[i, :],
                             color='grey', zorder=100)
                if idx == 0:
                    plt.ylabel('Visibility Squared')
                else:
                    ax.set_yticklabels([])
                plt.ylim(-0.03, 1.1)
                if nicer:
                    # ax.set_xticklabels([])
                    ax.set_xticks([])
                else:
                    plt.legend()
                    plt.xlabel('spatial frequency (1/arcsec)')
            plt.suptitle(title_name, y=0.92)
            plt.show()

        # T3
        if self.fit_for[2]:
            try:
                c1 = plotdata[0][1][6]*(1-plotdata[0][1][8])
                c2 = plotdata[1][1][6]*(1-plotdata[1][1][8])
                cmax = np.nanmax(np.abs(np.concatenate((c1, c2))))
                if cmax < 100:
                    cmax = cmax*1.5
                else:
                    cmax=180
            except:
                cmax=180
            plt.figure(figsize=(10, 5))
            if plotsplit:
                gs = gridspec.GridSpec(1, 2, wspace=0.05)
            for idx in range(nplot):
                if plotsplit:
                    ax = plt.subplot(gs[0, idx])
                closure = plotdata[idx][1][6]
                closure_error = plotdata[idx][1][7]
                closure_flag = plotdata[idx][1][8]
                model_closure_full = fitres[idx][2]
                for i in range(0,4):
                    plt.errorbar(magu_as_T3[i, :],
                                 closure[i, :]*(1-closure_flag)[i],
                                 closure_error[i, :]*(1-closure_flag)[i],
                                 color=self.colors_closure[i],
                                 ls='', lw=1, alpha=0.5, capsize=0)
                    plt.scatter(magu_as_T3[i, :],
                                closure[i, :]*(1-closure_flag)[i],
                                color=self.colors_closure[i],
                                alpha=0.5, label=self.closure_labels[i])
                    if nicer:
                        plt.text(magu_as_T3[i, :].mean(), -cmax*1.06,
                                 self.closure_labels[i],
                                 color=self.colors_closure[i],
                                 ha='center', va='center')
                    plt.plot(magu_as_T3_model[i, :], model_closure_full[i, :],
                             color='grey', zorder=100)
                if idx == 0:
                    plt.ylabel('Closure Phase (deg)')
                else:
                    ax.set_yticklabels([])
                plt.ylim(-cmax,cmax)
                if nicer:
                    # ax.set_xticklabels([])
                    ax.set_xticks([])
                else:
                    plt.legend()
                    plt.xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
        plt.suptitle(title_name, y=0.92)
        plt.show()

        # Visphi
        if self.fit_for[3]:
            try:
                c1 = plotdata[0][1][9]*(1-plotdata[0][1][11])
                c2 = plotdata[1][1][9]*(1-plotdata[1][1][11])
                cmax = np.nanmax(np.abs(np.concatenate((c1, c2))))
                if cmax < 100:
                    cmax = cmax*1.5
                else:
                    cmax=180
                print(cmax)
            except:
                cmax=180
            plt.figure(figsize=(10, 5))
            if plotsplit:
                gs = gridspec.GridSpec(1, 2, wspace=0.05)
            for idx in range(nplot):
                if plotsplit:
                    ax = plt.subplot(gs[0, idx])
                visphi = plotdata[idx][1][9]
                visphi_error = plotdata[idx][1][10]
                visphi_flag = plotdata[idx][1][11]
                model_visphi_full = fitres[idx][1]
                for i in range(0,6):
                    plt.errorbar(magu_as[i, :],
                                 visphi[i, :]*(1-visphi_flag)[i],
                                 visphi_error[i, :]*(1-visphi_flag)[i],
                                 color=self.colors_baseline[i],
                                 ls='', lw=1, alpha=0.5, capsize=0)
                    plt.scatter(magu_as[i,:],
                                visphi[i, :]*(1-visphi_flag)[i],
                                color=self.colors_baseline[i],
                                alpha=0.5, label=self.baseline_labels[i])
                    if nicer:
                        plt.text(magu_as[i, :].mean(), -cmax*1.06,
                                 self.baseline_labels[i],
                                 color=self.colors_baseline[i],
                                 ha='center', va='center')
                    plt.plot(magu_as_model[i, :], model_visphi_full[i, :],
                             color='grey', zorder=100)
                if idx == 0:
                    plt.ylabel('visibility phase')
                else:
                    ax.set_yticklabels([])
                plt.ylim(-cmax,cmax)
                if nicer:
                    # ax.set_xticklabels([])
                    ax.set_xticks([])
                else:
                    plt.legend()
                    plt.xlabel('spatial frequency (1/arcsec)')
            plt.suptitle(title_name, y=0.92)
            plt.show()

    # def plotFit(self, theta, fitdata, fitarg, fithelp, idx=0, createpdf=False):
    #     """
    #     Calculates the theoretical interferometric data for the given
    #     parameters in theta and plots them together with the data in fitdata.
    #     """
    #     rad2as = 180 / np.pi * 3600
    #     stname = self.name.find('GRAVI')
    #     title_name = self.name[stname:-5]
    # 
    #     (visamp, visamp_error, visamp_flag,
    #      vis2, vis2_error, vis2_flag,
    #      closure, closure_error, closure_flag,
    #      visphi, visphi_error, visphi_flag) = fitdata
    # 
    #     wave = self.wlSC
    #     dlambda = self.dlambda
    #     if self.phasemaps:
    #         wave_model = wave
    #     else:
    #         wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
    #     dlambda_model = np.zeros((6,len(wave_model)))
    #     for i in range(0,6):
    #         dlambda_model[i,:] = np.interp(wave_model, wave, dlambda[i,:])
    # 
    #     # Fit
    #     u = self.u
    #     v = self.v
    #     magu = np.sqrt(u**2.+v**2.)
    #     self.wave = wave_model
    #     self.dlambda = dlambda_model
    # 
    #     fithelp[4] = wave_model
    #     fithelp[5] = dlambda_model
    # 
    #     (model_visamp_full, model_visphi_full,
    #     model_closure_full)  = _calc_vis_mstars(theta, fitarg, fithelp)
    #     self.wave = wave
    #     self.dlambda = dlambda
    # 
    #     model_vis2_full = model_visamp_full**2.
    # 
    #     magu_as = self.spFrequAS
    # 
    #     u_as_model = np.zeros((len(u),len(wave_model)))
    #     v_as_model = np.zeros((len(v),len(wave_model)))
    #     for i in range(0,len(u)):
    #         u_as_model[i,:] = u[i]/(wave_model*1.e-6) / rad2as
    #         v_as_model[i,:] = v[i]/(wave_model*1.e-6) / rad2as
    #     magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)
    # 
    # 
    # 
    #     # Visamp
    #     if self.fit_for[0]:
    #         for i in range(0,6):
    #             plt.errorbar(magu_as[i,:], visamp[i,:]*(1-visamp_flag)[i],
    #                         visamp_error[i,:]*(1-visamp_flag)[i],
    #                         color=self.colors_baseline[i],ls='', lw=1, alpha=0.5, capsize=0)
    #             plt.scatter(magu_as[i,:], visamp[i,:]*(1-visamp_flag)[i],
    #                         color=self.colors_baseline[i], alpha=0.5, label=self.baseline_labels[i])
    #             plt.plot(magu_as_model[i,:], model_visamp_full[i,:],
    #                     color='k', zorder=100)
    #         plt.ylabel('visibility modulus')
    #         plt.ylim(-0.1,1.1)
    #         plt.xlabel('spatial frequency (1/arcsec)')
    #         plt.legend()
    #         if createpdf:
    #             savetime = self.savetime
    #             plt.title('Polarization %i' % (idx + 1))
    #             pdfname = '%s_pol%i_5.png' % (savetime, idx)
    #             plt.savefig(pdfname)
    #             plt.close()
    #         else:
    #             plt.title(title_name)
    #             plt.show()
    # 
    #     # Vis2
    #     if self.fit_for[1]:
    #         for i in range(0,6):
    #             plt.errorbar(magu_as[i,:], vis2[i,:]*(1-vis2_flag)[i],
    #                         vis2_error[i,:]*(1-vis2_flag)[i],
    #                         color=self.colors_baseline[i],ls='', lw=1, alpha=0.5, capsize=0)
    #             plt.scatter(magu_as[i,:], vis2[i,:]*(1-vis2_flag)[i],
    #                         color=self.colors_baseline[i],alpha=0.5, label=self.baseline_labels[i])
    #             plt.plot(magu_as_model[i,:], model_vis2_full[i,:],
    #                     color='k', zorder=100)
    #         plt.xlabel('spatial frequency (1/arcsec)')
    #         plt.ylabel('visibility squared')
    #         plt.ylim(-0.1,1.1)
    #         plt.legend()
    #         if createpdf:
    #             plt.title('Polarization %i' % (idx + 1))
    #             pdfname = '%s_pol%i_6.png' % (savetime, idx)
    #             plt.savefig(pdfname)
    #             plt.close()
    #         else:
    #             plt.title(title_name)
    #             plt.show()
    # 
    #     # T3
    #     if self.fit_for[2]:
    #         for i in range(0,4):
    #             max_u_as_model = self.max_spf[i]/(wave_model*1.e-6) / rad2as
    #             plt.errorbar(self.spFrequAS_T3[i,:], closure[i,:]*(1-closure_flag)[i],
    #                         closure_error[i,:]*(1-closure_flag)[i],
    #                         color=self.colors_closure[i],ls='', lw=1, alpha=0.5, capsize=0)
    #             plt.scatter(self.spFrequAS_T3[i,:], closure[i,:]*(1-closure_flag)[i],
    #                         color=self.colors_closure[i], alpha=0.5, label=self.closure_labels[i])
    #             plt.plot(max_u_as_model, model_closure_full[i,:],
    #                     color='k', zorder=100)
    #         plt.xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
    #         plt.ylabel('closure phase (deg)')
    #         plt.ylim(-180,180)
    #         plt.legend()
    #         if createpdf:
    #             plt.title('Polarization %i' % (idx + 1))
    #             pdfname = '%s_pol%i_7.png' % (savetime, idx)
    #             plt.savefig(pdfname)
    #             plt.close()
    #         else:
    #             plt.title(title_name)
    #             plt.show()
    # 
    #     # VisPhi
    #     if self.fit_for[3]:
    #         for i in range(0,6):
    #             plt.errorbar(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i],
    #                         visphi_error[i,:]*(1-visphi_flag)[i],
    #                         color=self.colors_baseline[i], ls='', lw=1, alpha=0.5, capsize=0)
    #             plt.scatter(magu_as[i,:], visphi[i,:]*(1-visphi_flag)[i],
    #                         color=self.colors_baseline[i], alpha=0.5, label=self.baseline_labels[i])
    #             plt.plot(magu_as_model[i,:], model_visphi_full[i,:],
    #                     color='k', zorder=100)
    #         plt.ylabel('visibility phase')
    #         plt.xlabel('spatial frequency (1/arcsec)')
    #         plt.legend()
    #         if createpdf:
    #             plt.title('Polarization %i' % (idx + 1))
    #             pdfname = '%s_pol%i_8.png' % (savetime, idx)
    #             plt.savefig(pdfname)
    #             plt.close()
    #         else:
    #             plt.title(title_name)
    #             plt.show()


def _lnprob_night(theta, fitdata, lower, upper, theta_names, fitarg, fithelp):
    lp = _lnprior_night(theta, lower, upper, theta_names)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _lnlike_night(theta, fitdata, fitarg, fithelp)


def _lnprior_night(theta, lower, upper, theta_names):
    if np.any(theta < lower) or np.any(theta > upper):
        return -np.inf
    gidx = [i for i, x in enumerate(theta_names) if 'coh' in x]
    lp = 0
    for idx in gidx:
        a = theta[idx]
        mu = 1
        sigma = 0.1
        lp += np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(a-mu)**2/sigma**2
    return lp


def _lnlike_night(theta, fitdata, fitarg, fithelp):
    (len_lightcurve, nsource, fit_for, bispec_ind, fit_mode,
     wave, dlambda, fixedBHalpha, oneBHalpha, oneBG, todel, fixed,
     phasemaps, pm_sources) = fithelp
    (visamp, visamp_error, visamp_flag,
     vis2, vis2_error, vis2_flag,
     closure, closure_error, closure_flag,
     visphi, visphi_error, visphi_flag) = fitdata

    for ddx in range(len(todel)):
        theta = np.insert(theta, todel[ddx], fixed[ddx])

    ln_prob_res = 0
    for ndx in range(len_lightcurve):
        _theta = np.zeros(nsource*3+10)
        for sdx in range(nsource):
            if sdx == 0:
                _theta[:2] = theta[:2]
            else:
                _theta[sdx*3-1] = theta[sdx*2]
                _theta[sdx*3] = theta[sdx*2+1]
                _theta[sdx*3+1] = theta[nsource*2+sdx-1]

            th_rest = nsource*3-1
            if oneBHalpha:
                _theta[th_rest] = theta[nsource*3-1]
            else:
                _theta[th_rest] = theta[nsource*3-1 + ndx*11]
            if oneBG:
                _theta[th_rest+1] = theta[nsource*3-1 + ndx*11 + 1]
            else:
                _theta[th_rest+1] = theta[nsource*3-1 + ndx*11 + 1]
            _theta[th_rest+2] = theta[nsource*3-1 + ndx*11 + 2]
            _theta[th_rest+3] = theta[nsource*3-1 + ndx*11 + 3]
            _theta[th_rest+4] = theta[nsource*3-1 + ndx*11 + 4]
            _theta[th_rest+5:] = theta[nsource*3-1 + ndx*11+5
                                       :nsource*3-1 + ndx*11+11]

        if phasemaps:
            _pm_sources = pm_sources[ndx]
            pm_amp_c, pm_pha_c, pm_int_c = _pm_sources[0]
            _pm_sources = _pm_sources[1:]
            _fithelp = [nsource, fit_for, bispec_ind, fit_mode,
                        wave, dlambda, fixedBHalpha, None, None, phasemaps,
                        None, None, None, None, None, None, False,
                        _pm_sources, pm_amp_c, pm_pha_c, pm_int_c]
        else:
            _fithelp = [nsource, fit_for, bispec_ind, fit_mode,
                        wave, dlambda, fixedBHalpha, None, None, phasemaps,
                        None, None, None, None, None, None, False,
                        None, None, None, None]

        (model_visamp, model_visphi,
         model_closure) = _calc_vis_mstars(_theta, fitarg[:, ndx], _fithelp)
        model_vis2 = model_visamp**2.

        #Data
        res_visamp = np.sum(-(model_visamp-visamp[ndx])**2
                            /visamp_error[ndx]**2*(1-visamp_flag[ndx]))
        res_vis2 = np.sum(-(model_vis2-vis2[ndx])**2.
                          /vis2_error[ndx]**2.*(1-vis2_flag[ndx]))

        res_closure = np.degrees(np.abs(np.exp(1j*np.radians(model_closure))
                                        - np.exp(1j*np.radians(closure[ndx]))))
        res_clos = np.sum(-res_closure**2./closure_error[ndx]**2.
                          * (1-closure_flag[ndx]))

        res_visphi = np.degrees(np.abs(np.exp(1j*np.radians(model_visphi))
                                       - np.exp(1j*np.radians(visphi[ndx]))))
        res_phi = np.sum(-res_visphi**2./visphi_error[ndx]**2.
                         * (1-visphi_flag[ndx]))

        ln_prob_res += 0.5 * (res_visamp * fit_for[0]
                              + res_vis2 * fit_for[1]
                              + res_clos * fit_for[2]
                              + res_phi * fit_for[3])
    return ln_prob_res




class GravMNightFit(GravNight):
    def __init__(self, file_list, verbose=False):
        super().__init__(file_list, verbose=verbose)

    def fitStars(self,
                 ra_list,
                 de_list,
                 fr_list=None,
                 fit_size=None,
                 fit_pos=None,
                 fit_fr=None,
                 nthreads=1,
                 nwalkers=301,
                 nruns=301,
                 bequiet=False,
                 fit_for=np.array([0.5, 0.5, 1.0, 0.0]),
                 error_scale=1,
                 fit_mode='numeric',
                 initial=None,
                 flagtill=3,
                 flagfrom=13,
                 fixedBHalpha=False,
                 oneBHalpha=False,
                 oneBG=True,
                 phasemaps=False,
                 interppm=True,
                 smoothkernel=15,
                 pmdatayear=2019):
        """
        Multi source fit to GRAVITY data
        Function fits a central source and a number of companion sources.
        All flux ratios are with respect to centra source

        The length of the input lists defines number of companions!

        Mandatory argumens:
        ra_list:        Initial guess for ra separation of companions
        de_list:        Initial guess for dec separation of companions
        fr_list:        Initial guess for flux ratio of companions

        Optional arguments for companions:
        If those vaues are given they need to be a list with one entry per companion
        fit_size:       Size of fitting area [5]
        fit_pos:        Fit position of each companion [True]
        fit_fr:         Fit flux ratio of each companion [True]

        Other optional arguments:
        nthreads:       number of cores [4]
        nwalkers:       number of walkers [500]
        nruns:          number of MCMC runs [500]
        bequiet:        Suppresses ALL outputs
        fit_for:        weight of VA, V2, T3, VP [[0.5,0.5,1.0,0.0]]
        fit_mode:       Kind of integration for visibilities
                        (approx, numeric, analytic) [numeric]
        initial:        Initial guess for fit [None]
        flagtill:       Flag blue channels, has to be changed for not LOW [3]
        flagfrom:       Flag red channels, has to be changed for not LOW [13]
        fixedBHalpha:   No fit for black hole power law [False]
        oneBHalpha:     One power law index for all files [False]
        phasemaps:      Use Phasemaps for fit [False]
        interppm:       Interpolate Phasemaps [True]
        smoothkernel:   Size of smoothing kernel in mas [15]
        """

        if (self.datalist[0].resolution != 'LOW'
                and flagtill == 3 and flagfrom == 13):
            raise ValueError('Initial values for flagtill and flagfrom have'
                             'to be changed if not low resolution')

        self.fit_for = fit_for
        self.fixedBHalpha = fixedBHalpha
        self.oneBHalpha = oneBHalpha
        self.oneBG = oneBG
        self.interppm = interppm
        self.fit_mode = fit_mode
        self.bequiet = bequiet
        self.nruns = nruns

        self.phasemaps = phasemaps
        self.datayear = pmdatayear
        self.smoothkernel = smoothkernel

        nsource = len(ra_list)
        nfiles = len(self.datalist)*2
        if fit_size is None:
            fit_size = np.ones(nsource)*5
        if fit_pos is None:
            fit_pos = np.ones(nsource)
        if fit_fr is None:
            fit_fr = np.ones(nsource-1)

        if len(de_list) != nsource or len(fit_size) != nsource:
            raise ValueError('list of input parameters have different lengths')
        if len(fit_pos) != nsource:
            raise ValueError('list of input parameters have different lengths')
        if fr_list is None:
            fr_list = np.ones(nsource-1)
        else:
            if len(fr_list) != (nsource-1):
                raise ValueError('list of fr_list has to be the'
                                 'same number of sources -1')

        self.nsource = nsource
        self.nfiles = nfiles

        nwave = self.datalist[0].channel
        for num, obj in enumerate(self.datalist):
            if obj.channel != nwave:
                raise ValueError('File number %i has different amount of '
                                 'channels' % num)

        MJD = []
        u, v = [], []
        for obj in self.datalist:
            obj.getIntdata(plot=False, flag=False)
            obj.getDlambda()

            MJD.append(fits.open(obj.name)[0].header["MJD-OBS"])
            u.append(obj.u)
            v.append(obj.v)

            if self.datalist[0].polmode == 'SPLIT':
                MJD.append(fits.open(obj.name)[0].header["MJD-OBS"])
                u.append(obj.u)
                v.append(obj.v)

            self.wave = obj.wlSC
            self.dlambda = obj.dlambda
            self.bispec_ind = obj.bispec_ind

        # Get data
        if self.polmode == 'SPLIT':
            visamp_P = []
            visamp_error_P = []
            visamp_flag_P = []

            vis2_P = []
            vis2_error_P = []
            vis2_flag_P = []

            closure_P = []
            closure_error_P = []
            closure_flag_P = []

            visphi_P = []
            visphi_error_P = []
            visphi_flag_P = []

            ndit = []

            for obj in self.datalist:
                visamp_P.append(obj.visampSC_P1)
                visamp_error_P.append(obj.visamperrSC_P1 * error_scale)
                visamp_flag_P.append(obj.visampflagSC_P1)
                visamp_P.append(obj.visampSC_P2)
                visamp_error_P.append(obj.visamperrSC_P2 * error_scale)
                visamp_flag_P.append(obj.visampflagSC_P2)

                vis2_P.append(obj.vis2SC_P1)
                vis2_error_P.append(obj.vis2errSC_P1 * error_scale)
                vis2_flag_P.append(obj.vis2flagSC_P1)
                vis2_P.append(obj.vis2SC_P2)
                vis2_error_P.append(obj.vis2errSC_P2 * error_scale)
                vis2_flag_P.append(obj.vis2flagSC_P2)

                closure_P.append(obj.t3SC_P1)
                closure_error_P.append(obj.t3errSC_P1 * error_scale)
                closure_flag_P.append(obj.t3flagSC_P1)
                closure_P.append(obj.t3SC_P2)
                closure_error_P.append(obj.t3errSC_P2 * error_scale)
                closure_flag_P.append(obj.t3flagSC_P2)

                visphi_P.append(obj.visphiSC_P1)
                visphi_error_P.append(obj.visphierrSC_P1 * error_scale)
                visphi_flag_P.append(obj.visampflagSC_P1)
                visphi_P.append(obj.visphiSC_P2)
                visphi_error_P.append(obj.visphierrSC_P2 * error_scale)
                visphi_flag_P.append(obj.visampflagSC_P2)

                ndit.append(np.shape(obj.visampSC_P1)[0]//6)
                if ndit[-1] != 1:
                    raise ValueError('Only maxframe reduced files can be used'
                                     'for full night fits!')

        elif self.polmode == 'COMBINED':
            raise ValueError("Sorry, only SPLIT is implemented at the moment")

        visamp_P = np.array(visamp_P)
        visamp_error_P = np.array(visamp_error_P)
        visamp_flag_P = np.array(visamp_flag_P)

        vis2_P = np.array(vis2_P)
        vis2_error_P = np.array(vis2_error_P)
        vis2_flag_P = np.array(vis2_flag_P)

        closure_P = np.array(closure_P)
        closure_error_P = np.array(closure_error_P)
        visphi_flag_P = np.array(visphi_flag_P)

        visphi_P = np.array(visphi_P)
        visphi_error_P = np.array(visphi_error_P)
        closure_flag_P = np.array(closure_flag_P)

        with np.errstate(invalid='ignore'):
            visamp_flag1 = (visamp_P > 1) | (visamp_P < 1.e-5)
        visamp_flag2 = np.isnan(visamp_P)
        visamp_flag_P = ((visamp_flag_P) | (visamp_flag1) | (visamp_flag2))
        visamp_P = np.nan_to_num(visamp_P)
        visamp_error_P[visamp_flag_P] = 1.

        with np.errstate(invalid='ignore'):
            vis2_flag1 = (vis2_P > 1) | (vis2_P < 1.e-5)
        vis2_flag2 = np.isnan(vis2_P)
        vis2_flag_P = ((vis2_flag_P) | (vis2_flag1) | (vis2_flag2))
        vis2_P = np.nan_to_num(vis2_P)
        vis2_error_P[vis2_flag_P] = 1.

        closure_P = np.nan_to_num(closure_P)
        visphi_P = np.nan_to_num(visphi_P)
        visphi_flag_P[np.where(visphi_error_P == 0)] = True
        visphi_error_P[np.where(visphi_error_P == 0)] = 100
        closure_flag_P[np.where(closure_error_P == 0)] = True
        closure_error_P[np.where(closure_error_P == 0)] = 100

        for num in range(nfiles):
            if ((flagtill > 0) and (flagfrom > 0)):
                p = flagtill
                t = flagfrom
                if num == 0 and not bequiet:
                    if not bequiet:
                        print('using channels from #%i to #%i' % (p, t))
                visamp_flag_P[num, :, 0:p] = True
                vis2_flag_P[num, :, 0:p] = True
                visphi_flag_P[num, :, 0:p] = True
                closure_flag_P[num, :, 0:p] = True

                visamp_flag_P[num, :, t:] = True
                vis2_flag_P[num, :, t:] = True
                visphi_flag_P[num, :, t:] = True
                closure_flag_P[num, :, t:] = True

        if initial is not None:
            if len(initial) != 5:
                raise ValueError('Length of initial parameter '
                                 'list is not correct, should be 4: '
                                 'fr_BH, fr BG, alpha, pc ra, pc dec')
            (fr_BH, flux_ratio_bg_in,
             alpha_SgrA_in, pc_RA_in, pc_DEC_in) = initial
        else:
            alpha_SgrA_in = -0.5
            flux_ratio_bg_in = 1
            pc_RA_in = 0
            pc_DEC_in = 0
            fr_BH = 0
        lightcurve_list = np.ones(nfiles)*fr_BH
        fluxBG_list = np.ones(nfiles)*flux_ratio_bg_in

        # nsource*2 positions
        # (nsource - 1) source flux ratios
        # len(files) * (flux ratio + bg + pc*2 + sgra color + 6 coherence loss)
        theta = np.zeros(nsource*2 + (nsource-1) + nfiles*11)
        lower = np.zeros(nsource*2 + (nsource-1) + nfiles*11)
        upper = np.zeros(nsource*2 + (nsource-1) + nfiles*11)
        todel = []
        theta_names = []
        pc_size = 5
        for ndx in range(nsource):
            theta[ndx*2] = ra_list[ndx]
            theta[ndx*2+1] = de_list[ndx]

            lower[ndx*2] = ra_list[ndx] - fit_size[ndx]
            lower[ndx*2+1] = de_list[ndx] - fit_size[ndx]

            upper[ndx*2] = ra_list[ndx] + fit_size[ndx]
            upper[ndx*2+1] = de_list[ndx] + fit_size[ndx]

            if not fit_pos[ndx]:
                todel.append(ndx*2)
                todel.append(ndx*2+1)
            theta_names.append('dRA%i' % (ndx + 1))
            theta_names.append('dDEC%i' % (ndx + 1))
            if not bequiet:
                if ndx == 0:
                    print('Initial conditions:')
                print('dRA%i    = %.2f' % ((ndx + 1), theta[ndx*2]))
                print('dDec%i   = %.2f' % ((ndx + 1), theta[ndx*2+1]))

        for ndx in range(nsource-1):
            theta[nsource*2+ndx] = np.log10(fr_list[ndx])
            lower[nsource*2+ndx] = np.log10(0.001)
            upper[nsource*2+ndx] = np.log10(100)
            if not fit_fr[ndx]:
                todel.append(nsource*2 + 4 + nfiles + ndx)
            theta_names.append('fr%i' % (ndx + 2))
            if not bequiet:
                print('fr %i/1  = %.2f' % ((ndx + 2), fr_list[ndx]))
        if not bequiet:
            print('fr BH/1 = %.2f' % (10**lightcurve_list[0]))
            print('fr BG   = %.2f' % (fluxBG_list[0]))
            print('alphaBH = %.2f' % (alpha_SgrA_in))

        for ndx in range(nfiles):
            theta[nsource*3-1 + ndx*11] = alpha_SgrA_in
            lower[nsource*3-1 + ndx*11] = -10
            upper[nsource*3-1 + ndx*11] = 10

            theta[nsource*3-1 + ndx*11 + 1] = fluxBG_list[ndx]
            lower[nsource*3-1 + ndx*11 + 1] = 0.1
            upper[nsource*3-1 + ndx*11 + 1] = 20

            theta[nsource*3-1 + ndx*11 + 2] = pc_RA_in
            lower[nsource*3-1 + ndx*11 + 2] = pc_RA_in - pc_size
            upper[nsource*3-1 + ndx*11 + 2] = pc_RA_in + pc_size

            theta[nsource*3-1 + ndx*11 + 3] = pc_DEC_in
            lower[nsource*3-1 + ndx*11 + 3] = pc_DEC_in - pc_size
            upper[nsource*3-1 + ndx*11 + 3] = pc_DEC_in + pc_size

            theta[nsource*3-1 + ndx*11 + 4] = lightcurve_list[ndx]
            lower[nsource*3-1 + ndx*11 + 4] = np.log10(0.001)
            upper[nsource*3-1 + ndx*11 + 4] = np.log10(100)

            theta[nsource*3-1 + ndx*11 + 5:nsource*3-1 + ndx*11+11] = 1.0
            lower[nsource*3-1 + ndx*11 + 5:nsource*3-1 + ndx*11+11] = 0.5
            upper[nsource*3-1 + ndx*11 + 5:nsource*3-1 + ndx*11+11] = 1.5

            theta_names.append('alphaBH%i' % (ndx+1))
            theta_names.append('frBG%i' % (ndx+1))
            theta_names.append('pcRa%i' % (ndx+1))
            theta_names.append('pcDec%i' % (ndx+1))
            theta_names.append('frBH%i' % (ndx+1))
            for cdx in range(6):
                theta_names.append('coh%i-%i' % ((cdx+1), (ndx+1)))
            if fixedBHalpha:
                todel.append(nsource*3-1 + ndx*11)
            if oneBHalpha and ndx > 0:
                todel.append(nsource*3-1 + ndx*11)
            # only one BG
            if oneBG and ndx > 0:
                todel.append(nsource*3-1 + ndx*11 + 1)
            if fit_for[3] == 0:
                todel.append(nsource*3-1 + ndx*11 + 2)
                todel.append(nsource*3-1 + ndx*11 + 3)
            # block coherence loss
            # todel.extend(np.arange(nsource*3-1 + ndx*11 + 5, nsource*3-1 + ndx*11+11))

        self.theta_in = np.copy(theta)
        self.theta_allnames = np.copy(theta_names)
        self.theta_names = theta_names
        ndim = len(theta)
        self.ndof = ndim - len(todel)

        if self.phasemaps:
            phasemaps = GravPhaseMaps()
            phasemaps.tel = self.tel
            phasemaps.resolution = self.resolution
            phasemaps.smoothkernel = self.smoothkernel
            phasemaps.datayear = self.datayear
            phasemaps.wlSC = self.datalist[0].wlSC
            phasemaps.interppm = interppm
            phasemaps.loadPhasemaps(interp=interppm)

            header = self.headerlist[0]
            northangle1 = header['ESO QC ACQ FIELD1 NORTH_ANGLE']/180*math.pi
            northangle2 = header['ESO QC ACQ FIELD2 NORTH_ANGLE']/180*math.pi
            northangle3 = header['ESO QC ACQ FIELD3 NORTH_ANGLE']/180*math.pi
            northangle4 = header['ESO QC ACQ FIELD4 NORTH_ANGLE']/180*math.pi
            self.northangle = [northangle1, northangle2,
                               northangle3, northangle4]
            ddec = []
            dra = []
            for header in self.headerlist:
                ddec1 = header['ESO QC MET SOBJ DDEC1']
                ddec2 = header['ESO QC MET SOBJ DDEC2']
                ddec3 = header['ESO QC MET SOBJ DDEC3']
                ddec4 = header['ESO QC MET SOBJ DDEC4']
                ddec.append([ddec1, ddec2, ddec3, ddec4])

                dra1 = header['ESO QC MET SOBJ DRA1']
                dra2 = header['ESO QC MET SOBJ DRA2']
                dra3 = header['ESO QC MET SOBJ DRA3']
                dra4 = header['ESO QC MET SOBJ DRA4']
                dra.append([dra1, dra2, dra3, dra4])

            pm_sources = []
            for ndx in range(nfiles):
                _sources = []
                pm_amp, pm_pha, pm_int = phasemaps.phasemap_source(0, 0,
                                                              self.northangle,
                                                              dra[ndx//2],
                                                              ddec[ndx//2])
                _sources.append([pm_amp, pm_pha, pm_int])
                for sdx in range(nsource):
                    pm_amp, pm_pha, pm_int = phasemaps.phasemap_source(theta[sdx*2],
                                                                  theta[sdx*2+1],
                                                                  self.northangle,
                                                                  dra[ndx//2],
                                                                  ddec[ndx//2])
                    _sources.append([pm_amp, pm_pha, pm_int])
                pm_sources.append(_sources)

        for ddx in sorted(todel, reverse=True):
            del theta_names[ddx]
        fixed = theta[todel]
        theta = np.delete(theta, todel)
        lower = np.delete(lower, todel)
        upper = np.delete(upper, todel)
        self.fixed = fixed
        self.todel = todel

        ndim = len(theta)
        width = 1e-1
        pos = np.ones((nwalkers, ndim))
        for par in range(ndim):
            if 'coh' in theta_names:
                pos[:, par] = theta[par] + width*5e-2*np.random.randn(nwalkers)
            else:
                pos[:, par] = theta[par] + width*np.random.randn(nwalkers)
        self.todel = todel
        self.ndim = ndim

        fitdata = [visamp_P, visamp_error_P, visamp_flag_P,
                   vis2_P, vis2_error_P, vis2_flag_P,
                   closure_P, closure_error_P, closure_flag_P,
                   visphi_P, visphi_error_P, visphi_flag_P]

        fitarg = np.array([u, v])
        if self.phasemaps:
            fithelp = [self.nfiles, self.nsource, self.fit_for,
                       self.bispec_ind, self.fit_mode, self.wave,
                       self.dlambda, self.fixedBHalpha, oneBHalpha,
                       oneBG, todel, fixed, self.phasemaps, pm_sources]
        else:
            fithelp = [self.nfiles, self.nsource, self.fit_for,
                       self.bispec_ind, self.fit_mode, self.wave,
                       self.dlambda, self.fixedBHalpha, oneBHalpha,
                       oneBG, todel, fixed, self.phasemaps, None]

        # _lnprob_night(theta, fitdata, lower, upper, theta_names,
        #               fitarg, fithelp)
        if nthreads == 1:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, _lnprob_night,
                                                 args=(fitdata, lower,
                                                       upper, theta_names,
                                                       fitarg, fithelp))
            if bequiet:
                self.sampler.run_mcmc(pos, nruns, progress=False)
            else:
                self.sampler.run_mcmc(pos, nruns, progress=True)
        else:
            with Pool(processes=nthreads) as pool:
                self.sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                                     _lnprob_night,
                                                     args=(fitdata, lower,
                                                           upper, theta_names,
                                                           fitarg,
                                                           fithelp),
                                                     pool=pool)
                if bequiet:
                    self.sampler.run_mcmc(pos, nruns, progress=False)
                else:
                    self.sampler.run_mcmc(pos, nruns, progress=True)

        self.fitdata = fitdata
        self.fithelp = fithelp
        self.fitarg = fitarg
        self.MJD = MJD

    def getFitResult(self, plot=True, plotcorner=False, ret=False):
        samples = self.sampler.chain
        self.mostprop = self.sampler.flatchain[np.argmax(self.sampler.flatlnprobability)]
        print("Mean acceptance fraction: %.2f"
              % np.mean(self.sampler.acceptance_fraction))

        clinitial = np.delete(self.theta_in, self.todel)
        clsamples = samples  # np.delete(samples, self.todel, 2)
        cllabels = self.theta_names  # np.delete(self.theta_names, self.todel)
        clmostprop = self.mostprop  # np.delete(self.mostprop, self.todel)
        cldim = len(clmostprop)

        if self.nruns > 300:
            fl_samples = samples[:, -200:, :].reshape((-1, self.ndim))
            fl_clsamples = clsamples[:, -200:, :].reshape((-1, cldim))
        elif self.nruns > 200:
            fl_samples = samples[:, -100:, :].reshape((-1, self.ndim))
            fl_clsamples = clsamples[:, -100:, :].reshape((-1, cldim))
        else:
            fl_samples = samples.reshape((-1, self.ndim))
            fl_clsamples = clsamples.reshape((-1, cldim))
        self.fl_clsamples = fl_clsamples
        self.medianprop = np.percentile(fl_samples, [50],axis=0)[0]

        percentiles = np.percentile(fl_clsamples, [16, 50, 84], axis=0).T
        percentiles[:, 0] = percentiles[:,1] - percentiles[:,0]
        percentiles[:, 2] = percentiles[:,2] - percentiles[:,1]

        fittab = pd.DataFrame()
        fittab["column"] = ["in", "M.L.", "M.P.", "$-\sigma$", "$+\sigma$"]
        _ct_del = 0
        _ct_used = 0
        for idx, name in enumerate(self.theta_allnames):
            if idx in self.todel:
                fittab[name] = pd.Series([self.fixed[_ct_del],
                                          self.fixed[_ct_del],
                                          self.fixed[_ct_del],
                                          0,
                                          0])
                _ct_del += 1
            else:
                fittab[name] = pd.Series([clinitial[_ct_used],
                                          clmostprop[_ct_used],
                                          percentiles[_ct_used, 1],
                                          percentiles[_ct_used, 0],
                                          percentiles[_ct_used, 2]])
                _ct_used += 1

        len_lightcurve = self.nfiles
        _lightcurve_all = 10**(clmostprop[-(len_lightcurve+self.nsource-1):-(self.nsource-1)])
        self.lightcurve = np.array([_lightcurve_all[::2],_lightcurve_all[1::2]])
        self.fitres = clmostprop
        self.fittab = fittab
        keys = fittab.keys()
        cohkeys = [x for x in keys if 'coh' in x]
        self.fittab_short = fittab.drop(columns=cohkeys)


        if plot:
            self.plot_MCMC(plotcorner)

        if ret:
            return self.medianprop

    def plot_MCMC(self, plotcorner=False):
        # clsamples = np.delete(self.sampler.chain, self.todel, 2)
        # cllabels = np.delete(self.theta_names, self.todel)
        # clmostprop = np.delete(self.mostprop, self.todel)
        clsamples = self.sampler.chain  # np.delete(self.sampler.chain, self.todel, 2)
        cllabels = self.theta_names  # np.delete(self.theta_names, self.todel)
        clmostprop = self.mostprop  # np.delete(self.mostprop, self.todel)
        cldim = len(clmostprop)

        fig, axes = plt.subplots(cldim, figsize=(8, cldim/1.5),
                                 sharex=True)
        for i in range(cldim):
            ax = axes[i]
            ax.plot(clsamples[:, :, i].T, "k", alpha=0.3)
            ax.axhline(clmostprop[i], color='C0', alpha=0.5)
            ax.set_ylabel(cllabels[i], rotation=0)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.show()

        if plotcorner:
            ranges = np.percentile(self.fl_clsamples, [3, 97], axis=0).T
            fig = corner.corner(self.fl_clsamples, quantiles=[0.16, 0.5, 0.84],
                                truths=clmostprop, labels=cllabels)
            plt.show()

    # def plot_fit_better(self, fitdata, fitres, fitarg, fithelp, figsize=None):
    #     (len_lightcurve, nsource, fit_for, bispec_ind, fit_mode,
    #     wave, dlambda, fixedBHalpha, soh_loss, phasemaps,
    #     northA, amp_map_int, pha_map_int, amp_map_denom_int, wave) = fithelp
    # 
    #     (visamp_d, visamp_error_d, visamp_flag_d,
    #      vis2_d, vis2_error_d, vis2_flag_d,
    #      closure_d, closure_error_d, closure_flag_d,
    #      visphi_d, visphi_error_d, visphi_flag_d) = fitdata
    #     if figsize is None:
    #         plt.figure(figsize=(7,len_lightcurve*0.7))
    #     else:
    #         plt.figure(figsize=figsize)
    #     gs = gridspec.GridSpec(len_lightcurve//2, 3, hspace=0.05)
    #     for idx in range(len_lightcurve//2):
    #         num = idx*2
    #         obj = self.gravData_list[idx]
    # 
    #         theta_ = np.copy(fitres[:nsource*2 + 4])
    #         theta_ = np.append(theta_, fitres[num + len(theta_)])
    #         theta_ = np.append(theta_, fitres[-nsource+1:])
    #         visamp, visphi, closure = _calc_vis(theta_, fitarg[:, num], fithelp)
    # 
    #         theta_ = np.copy(fitres[:nsource*2 + 4])
    #         theta_ = np.append(theta_, fitres[num+1 + len(theta_)])
    #         theta_ = np.append(theta_, fitres[-nsource+1:])
    #         visamp2, visphi2, closure2 = _calc_vis(theta_, fitarg[:, num+1], fithelp)
    # 
    # 
    #         vis2 = visamp**2
    #         vis22 = visamp2**2
    #         magu_as = obj.spFrequAS
    # 
    #         ax = plt.subplot(gs[idx,0])
    #         for i in range(6):
    #             plt.errorbar(magu_as[i,:], visamp_d[num][i,:]*(1-visamp_flag_d[num])[i],
    #                         visamp_error_d[num][i,:]*(1-visamp_flag_d[num])[i],
    #                         color=obj.colors_baseline[i], ls='', marker='o', markersize=2,
    #                         lw=1, alpha=1, capsize=0)
    # 
    #             plt.errorbar(magu_as[i,:], visamp_d[num+1][i,:]*(1-visamp_flag_d[num+1])[i],
    #                         visamp_error_d[num+1][i,:]*(1-visamp_flag_d[num+1])[i],
    #                         color=obj.colors_baseline[i], ls='', marker='o', markersize=2,
    #                         lw=1, alpha=1, capsize=0)
    # 
    #             plt.plot(magu_as[i, :], visamp[i, :], color='k', zorder=100)
    #             plt.plot(magu_as[i, :], visamp2[i, :], ls='--', color='k', zorder=100)
    #         plt.ylim(0,1.1)
    #         ax.set_xticklabels([])
    #         if idx == 0:
    #             plt.title('Visamp')
    #         plt.ylabel('File %i' % (idx+1))
    # 
    #         ax = plt.subplot(gs[idx,1])
    #         for i in range(6):
    #             plt.errorbar(magu_as[i,:], vis2_d[num][i,:]*(1-vis2_flag_d[num])[i],
    #                         vis2_error_d[num][i,:]*(1-vis2_flag_d[num])[i],
    #                         color=obj.colors_baseline[i], ls='', marker='o', markersize=2,
    #                         lw=1, alpha=1, capsize=0)
    # 
    #             plt.errorbar(magu_as[i,:], vis2_d[num+1][i,:]*(1-vis2_flag_d[num+1])[i],
    #                         vis2_error_d[num+1][i,:]*(1-vis2_flag_d[num+1])[i],
    #                         color=obj.colors_baseline[i], ls='', marker='o', markersize=2,
    #                         lw=1, alpha=1, capsize=0)
    # 
    #             plt.plot(magu_as[i, :], vis2[i, :], color='k', zorder=100)
    #             plt.plot(magu_as[i, :], vis22[i, :], ls='--', color='k', zorder=100)
    #         plt.ylim(0,1.1)
    #         ax.set_xticklabels([])
    #         if idx == 0:
    #             plt.title('Vis2')
    # 
    #         ax = plt.subplot(gs[idx,2])
    #         for i in range(4):
    #             plt.errorbar(obj.spFrequAS_T3[i,:], closure_d[num][i,:]*(1-closure_flag_d[num])[i],
    #                         closure_error_d[num][i,:]*(1-closure_flag_d[num])[i],
    #                         color=obj.colors_closure[i], ls='', marker='o', markersize=2,
    #                         lw=1, alpha=1, capsize=0)
    # 
    #             plt.errorbar(obj.spFrequAS_T3[i,:], closure_d[num+1][i,:]*(1-closure_flag_d[num+1])[i],
    #                         closure_error_d[num+1][i,:]*(1-closure_flag_d[num+1])[i],
    #                         color=obj.colors_closure[i], ls='', marker='o', markersize=2,
    #                         lw=1, alpha=1, capsize=0)
    # 
    #             plt.plot(obj.spFrequAS_T3[i,:], closure[i, :], color='k', zorder=100)
    #             plt.plot(obj.spFrequAS_T3[i,:], closure2[i, :], ls='--', color='k', zorder=100)
    #         plt.ylim(-180,180)
    #         ax.set_xticklabels([])
    #         if idx == 0:
    #             plt.title('Closure Phase')
    #     plt.show()
    # 
    # 
    # 
    # 
    # 
    # def plot_fit(self, fitres, fitarg, fithelp, axes=None):
    #     (len_lightcurve, nsource, fit_for, bispec_ind, fit_mode,
    #      wave, dlambda, fixedBHalpha, oneBHalpha, phasemaps, pm_sources) = fithelp
    #     if axes is None:
    #         #fig, axes = plt.subplots(2, 2, gridspec_kw={})
    #         fig, axes0 = plt.subplots()
    #         fig, axes1 = plt.subplots()
    #         fig, axes2 = plt.subplots()
    #         fig, axes3 = plt.subplots()
    #         axes = np.ones((2,2), dtype=object)
    # 
    #         axes[0,0] = axes0
    #         axes[0,1] = axes1
    #         axes[1,0] = axes2
    #         axes[1,1] = axes3
    # 
    #     for ndx in range(len_lightcurve):
    #         obj = self.datalist[ndx//2]
    #         _theta = np.zeros(nsource*3+10)
    #         for sdx in range(nsource):
    #             if sdx == 0:
    #                 _theta[:2] = fitres[:2]
    #             else:
    #                 _theta[sdx*3-1] = fitres[sdx*2]
    #                 _theta[sdx*3] = fitres[sdx*2+1]
    #                 _theta[sdx*3+1] = fitres[nsource*2+sdx-1]
    # 
    #         th_rest = nsource*3-1
    #         if oneBHalpha:
    #             _theta[th_rest] = fitres[nsource*3-1 + 1]
    #         else:
    #             _theta[th_rest] = fitres[nsource*3-1 + ndx*10+1]
    #         _theta[th_rest+1] = 0
    #         _theta[th_rest+2] = fitres[nsource*3-1 + ndx*10+2]
    #         _theta[th_rest+3] = fitres[nsource*3-1 + ndx*10+3]
    #         _theta[th_rest+4] = fitres[nsource*3-1 + ndx*10]
    #         _theta[th_rest+5:] = fitres[nsource*3-1 + ndx*10+4:nsource*3-1 + ndx*10+10]
    # 
    #         if phasemaps:
    #             _pm_sources = pm_sources[ndx]
    #             pm_amp_c, pm_pha_c, pm_int_c = _pm_sources[0]
    #             _pm_sources = _pm_sources[1:]
    #             _fithelp = [nsource, fit_for, bispec_ind, fit_mode,
    #                         wave, dlambda, fixedBHalpha, True, phasemaps,
    #                         None, None, None, None, None, None, False,
    #                         _pm_sources, pm_amp_c, pm_pha_c, pm_int_c]
    #         else:
    #             _fithelp = [nsource, fit_for, bispec_ind, fit_mode,
    #                         wave, dlambda, fixedBHalpha, True, phasemaps,
    #                         None, None, None, None, None, None, False,
    #                         None, None, None, None]
    #         visamp, visphi, closure = _calc_vis_mstars(_theta, fitarg[:, ndx],
    #                                                    _fithelp)
    #         visamp2 = visamp**2
    #         magu_as = obj.spFrequAS
    #         for i in range(6):
    #             axes[0,0].plot(magu_as[i, :], visamp[i, :], color='k', zorder=100)
    #             axes[0,1].plot(magu_as[i, :], visamp2[i, :], color='k', zorder=100)
    #             axes[1,1].plot(magu_as[i, :], visphi[i, :], color='k', zorder=100)
    #         for i in range(4):
    #             axes[1,0].plot(obj.spFrequAS_T3[i,:], closure[i, :], color='k', zorder=100)
    # 
    # def plot_data(self, fitdata, fitarg, axes=None):
    #     len_lightcurve = self.nfiles
    #     if axes is None:
    #         #fig, axes = plt.subplots(2, 2, gridspec_kw={})
    #         fig, axes0 = plt.subplots()
    #         fig, axes1 = plt.subplots()
    #         fig, axes2 = plt.subplots()
    #         fig, axes3 = plt.subplots()
    #         axes = np.ones((2,2), dtype=object)
    # 
    #         axes[0,0] = axes0
    #         axes[0,1] = axes1
    #         axes[1,0] = axes2
    #         axes[1,1] = axes3
    #     (visamp, visamp_error, visamp_flag,
    #         vis2, vis2_error, vis2_flag,
    #         closure, closure_error, closure_flag,
    #         visphi, visphi_error, visphi_flag) = fitdata
    #     (u, v) = fitarg
    #     for num in range(len_lightcurve):
    #         obj = self.datalist[num//2]
    #         magu_as = obj.spFrequAS
    #         for i in range(0,6):
    #             ## visamp
    #             axes[0,0].errorbar(magu_as[i,:], visamp[num][i,:]*(1-visamp_flag[num])[i],
    #                         visamp_error[num][i,:]*(1-visamp_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), ls='', lw=1, alpha=0.5, capsize=0)
    #             if num == 0:
    #                 axes[0,0].scatter(magu_as[i,:], visamp[num][i,:]*(1-visamp_flag[num])[i],
    #                             color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), alpha=0.5, label=obj.baseline_labels[i])
    #             else:
    #                 axes[0,0].scatter(magu_as[i,:], visamp[num][i,:]*(1-visamp_flag[num])[i],
    #                             color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), alpha=0.5)
    # 
    #             ## vis2
    #             axes[0,1].errorbar(magu_as[i,:], vis2[num][i,:]*(1-vis2_flag[num])[i],
    #                         vis2_error[num][i,:]*(1-vis2_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve),ls='', lw=1, alpha=0.5, capsize=0)
    # 
    #             axes[0,1].scatter(magu_as[i,:], vis2[num][i,:]*(1-vis2_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve),alpha=0.5)
    # 
    # 
    #             ## visphi
    # 
    #             axes[1,1].errorbar(magu_as[i,:], visphi[num][i,:]*(1-visphi_flag[num])[i],
    #                         visphi_error[num][i,:]*(1-visphi_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), ls='', lw=1, alpha=0.5, capsize=0)
    #             axes[1,1].scatter(magu_as[i,:], visphi[num][i,:]*(1-visphi_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), alpha=0.5)
    # 
    #         for i in range(4):
    #             ## t3visphi
    #             axes[1,0].errorbar(obj.spFrequAS_T3[i,:], closure[num][i,:]*(1-closure_flag[num])[i],
    #                         closure_error[num][i,:]*(1-closure_flag[num])[i],
    #                         color=lighten_color(obj.colors_closure[i], (num+1)/len_lightcurve),ls='', lw=1, alpha=0.5, capsize=0)
    #             if num == 0:
    #                 axes[1,0].scatter(obj.spFrequAS_T3[i,:], closure[num][i,:]*(1-closure_flag[num])[i],
    #                                   label=obj.closure_labels[i], color=lighten_color(obj.colors_closure[i], (num+1)/len_lightcurve))
    #             else:
    #                 axes[1,0].scatter(obj.spFrequAS_T3[i,:], closure[num][i,:]*(1-closure_flag[num])[i],
    #                                   color=lighten_color(obj.colors_closure[i], (num+1)/len_lightcurve))
    # 
    #     axes[0,0].set_ylabel('visibility modulus')
    #     axes[0,0].set_ylim(-0.1,1.1)
    #     axes[0,0].set_xlabel('spatial frequency (1/arcsec)')
    #     axes[0,0].legend(fontsize=12)
    # 
    #     axes[0,1].set_xlabel('spatial frequency (1/arcsec)')
    #     axes[0,1].set_ylabel('visibility squared')
    #     axes[0,1].set_ylim(-0.1,1.1)
    # 
    # 
    #     axes[1,0].set_xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
    #     axes[1,0].set_ylabel('closure phase (deg)')
    #     axes[1,0].set_ylim(-180,180)
    #     axes[1,0].legend(fontsize=12)
    # 
    #     axes[1,1].set_ylabel('visibility phase')
    #     axes[1,0].set_ylim(-180,180)
    #     axes[1,1].set_xlabel('spatial frequency (1/arcsec)')
    # 
    # def plot_residual(self, fitdata, fitarg, fitres, fithelp, axes=None, show=False):
    #     (len_lightcurve, nsource, fit_for, bispec_ind, fit_mode,
    #      wave, dlambda, fixedBHalpha, coh_loss, phasemaps, northA,
    #      amp_map_int, pha_map_int, amp_map_denom_int, wave) = fithelp
    #     if axes is None:
    #         #fig, axes = plt.subplots(2, 2, gridspec_kw={})
    #         fig, axes0 = plt.subplots()
    #         fig, axes1 = plt.subplots()
    #         fig, axes2 = plt.subplots()
    #         fig, axes3 = plt.subplots()
    #         axes = np.ones((2,2), dtype=object)
    # 
    #         axes[0,0] = axes0
    #         axes[0,1] = axes1
    #         axes[1,0] = axes2
    #         axes[1,1] = axes3
    #     (visamp, visamp_error, visamp_flag,
    #         vis2, vis2_error, vis2_flag,
    #         closure, closure_error, closure_flag,
    #         visphi, visphi_error, visphi_flag) = fitdata
    #     (u, v) = fitarg
    #     for num in range(len_lightcurve):
    #         obj = self.gravData_list[num//2]
    #         magu_as = obj.spFrequAS
    # 
    #         theta_ = np.copy(fitres[:nsource*2 + 4])
    #         theta_ = np.append(theta_,  fitres[num + nsource*2 + 4])
    #         theta_ = np.append(theta_,  fitres[-nsource+1:])
    # 
    #         model_visamp, model_visphi, model_closure = _calc_vis(theta_, fitarg[:, num], fithelp)
    #         model_visamp2 = model_visamp**2
    # 
    #         for i in range(0,6):
    #             ## visamp
    #             axes[0,0].errorbar(magu_as[i,:], (visamp-model_visamp)[num][i,:]*(1-visamp_flag[num])[i],
    #                         visamp_error[num][i,:]*(1-visamp_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), ls='', lw=1, alpha=0.5, capsize=0)
    #             if num == len_lightcurve-1:
    #                 axes[0,0].scatter(magu_as[i,:], (visamp-model_visamp)[num][i,:]*(1-visamp_flag[num])[i],
    #                             color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), alpha=0.5, label=obj.baseline_labels[i])
    #             else:
    #                 axes[0,0].scatter(magu_as[i,:], (visamp-model_visamp)[num][i,:]*(1-visamp_flag[num])[i],
    #                             color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), alpha=0.5)
    # 
    #             ## vis2
    #             axes[0,1].errorbar(magu_as[i,:], (vis2-model_visamp2)[num][i,:]*(1-vis2_flag[num])[i],
    #                         vis2_error[num][i,:]*(1-vis2_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve),ls='', lw=1, alpha=0.5, capsize=0)
    # 
    #             axes[0,1].scatter(magu_as[i,:], (vis2-model_visamp2)[num][i,:]*(1-vis2_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve),alpha=0.5)
    # 
    # 
    #             ## visphi
    # 
    #             axes[1,1].errorbar(magu_as[i,:], (visphi-model_visphi)[num][i,:]*(1-visphi_flag[num])[i],
    #                         visphi_error[num][i,:]*(1-visphi_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), ls='', lw=1, alpha=0.5, capsize=0)
    #             axes[1,1].scatter(magu_as[i,:], (visphi-model_visphi)[num][i,:]*(1-visphi_flag[num])[i],
    #                         color=lighten_color(obj.colors_baseline[i], (num+1)/len_lightcurve), alpha=0.5)
    # 
    #         for i in range(4):
    #             ## t3visphi
    #             axes[1,0].errorbar(obj.spFrequAS_T3[i,:], (closure-model_closure)[num][i,:]*(1-closure_flag[num])[i],
    #                         closure_error[num][i,:]*(1-closure_flag[num])[i],
    #                         color=lighten_color(obj.colors_closure[i], (num+1)/len_lightcurve),ls='', lw=1, alpha=0.5, capsize=0)
    #             if num == len_lightcurve-1:
    #                 axes[1,0].scatter(obj.spFrequAS_T3[i,:], (closure-model_closure)[num][i,:]*(1-closure_flag[num])[i],
    #                                   label=obj.closure_labels[i], color=lighten_color(obj.colors_closure[i], (num+1)/len_lightcurve))
    #             else:
    #                 axes[1,0].scatter(obj.spFrequAS_T3[i,:], (closure-model_closure)[num][i,:]*(1-closure_flag[num])[i],
    #                                   color=lighten_color(obj.colors_closure[i], (num+1)/len_lightcurve))
    # 
    #     axes[0,0].set_ylabel('visibility modulus')
    #     axes[0,0].set_ylim(-0.4,0.4)
    #     axes[0,0].set_xlabel('spatial frequency (1/arcsec)')
    #     axes[0,0].legend(fontsize=12)
    # 
    #     axes[0,1].set_xlabel('spatial frequency (1/arcsec)')
    #     axes[0,1].set_ylabel('visibility squared')
    #     axes[0,1].set_ylim(-0.4,0.4)
    # 
    # 
    #     axes[1,0].set_xlabel('spatial frequency of largest baseline in triangle (1/arcsec)')
    #     axes[1,0].set_ylabel('closure phase (deg)')
    #     axes[1,0].set_ylim(-40,40)
    #     axes[1,0].legend(fontsize=12)
    # 
    #     axes[1,1].set_ylabel('visibility phase')
    #     axes[1,1].set_ylim(-40,40)
    #     axes[1,1].set_xlabel('spatial frequency (1/arcsec)')
    #     if show:
    #         plt.show()
    # 
    # def plotFit(self):
    #     fig, axes0 = plt.subplots()
    #     fig, axes1 = plt.subplots()
    #     fig, axes2 = plt.subplots()
    #     fig, axes3 = plt.subplots()
    #     axes = np.ones((2,2), dtype=object)
    # 
    #     axes[0,0] = axes0
    #     axes[0,1] = axes1
    #     axes[1,0] = axes2
    #     axes[1,1] = axes3
    #     try:
    #         self.mostprop
    #         self.fitarg
    #         self.fithelp
    #         self.fitdata
    #         self.fitarg
    #     except NameError:
    #         print('mostprop, fitarg, fithelp and fitarg are not attributes. '
    #               'Need to run the fit first!')
    #     self.plot_fit(self.mostprop, self.fitarg, self.fithelp, axes=axes)
    #     self.plot_data(self.fitdata, self.fitarg, axes=axes)

    def getFitVis(self, fitres, fitarg, fithelp):
        (len_lightcurve, nsource, fit_for, bispec_ind, fit_mode,
         wave, dlambda, fixedBHalpha, oneBHalpha, oneBG, todel, fixed,
         phasemaps, pm_sources) = fithelp

        for ddx in range(len(todel)):
            fitres = np.insert(fitres, todel[ddx], fixed[ddx])

        allfitres = []
        for ndx in range(len_lightcurve):
            _theta = np.zeros(nsource*3+10)
            for sdx in range(nsource):
                if sdx == 0:
                    _theta[:2] = fitres[:2]
                else:
                    _theta[sdx*3-1] = fitres[sdx*2]
                    _theta[sdx*3] = fitres[sdx*2+1]
                    _theta[sdx*3+1] = fitres[nsource*2+sdx-1]

            th_rest = nsource*3-1
            if oneBHalpha:
                _theta[th_rest] = fitres[nsource*3-1]
            else:
                _theta[th_rest] = fitres[nsource*3-1 + ndx*11]
            if oneBG:
                _theta[th_rest+1] = fitres[nsource*3-1 + 1]
            else:
                _theta[th_rest+1] = fitres[nsource*3-1 + ndx*11 + 1]
            _theta[th_rest+2] = fitres[nsource*3-1 + ndx*11 + 2]
            _theta[th_rest+3] = fitres[nsource*3-1 + ndx*11 + 3]
            _theta[th_rest+4] = fitres[nsource*3-1 + ndx*11 + 4]
            _theta[th_rest+5:] = fitres[nsource*3-1 + ndx*11+5
                                        :nsource*3-1 + ndx*11+11]

            if phasemaps:
                _pm_sources = pm_sources[ndx]
                pm_amp_c, pm_pha_c, pm_int_c = _pm_sources[0]
                _pm_sources = _pm_sources[1:]
                _fithelp = [nsource, fit_for, bispec_ind, fit_mode,
                            wave, dlambda, fixedBHalpha, None, None, phasemaps,
                            None, None, None, None, None, None, False,
                            _pm_sources, pm_amp_c, pm_pha_c, pm_int_c]
            else:
                _fithelp = [nsource, fit_for, bispec_ind, fit_mode,
                            wave, dlambda, fixedBHalpha, None, None, phasemaps,
                            None, None, None, None, None, None, False,
                            None, None, None, None]
            (visamp, visphi,
             closure) = _calc_vis_mstars(_theta, fitarg[:, ndx],
                                                         _fithelp)
            visamp2 = visamp**2
            allfitres.append([visamp, visamp2, closure, visphi])
        return allfitres

    def plotFitComb(self, mostprop=True, nicer=True):
        rad2as = 180 / np.pi * 3600
        len_lightcurve = self.nfiles
        if mostprop:
            result = self.mostprop
        else:
            result = self.medianprop
        (visamp, visamp_error, visamp_flag,
         vis2, vis2_error, vis2_flag,
         closure, closure_error, closure_flag,
         visphi, visphi_error, visphi_flag) = self.fitdata
        (uu, vv) = self.fitarg

        wave = self.wave
        dlambda = self.dlambda
        if self.phasemaps:
            wave_model = wave
        else:
            wave_model = np.linspace(wave[0], wave[len(wave)-1], 1000)
        dlambda_model = np.zeros((6, len(wave_model)))
        for i in range(0, 6):
            dlambda_model[i, :] = np.interp(wave_model, wave, dlambda[i, :])

        fithelp_model = self.fithelp
        fithelp_model[5] = wave_model
        fithelp_model[6] = dlambda_model
        allfitres = self.getFitVis(result, self.fitarg, fithelp_model)

        obj = self.datalist[0]
        plot_quant = ['Visamp', 'Vis2', 'Closure Phase', 'Visibility Phase']
        plot_closure = [0, 0, 1, 0]
        plot_min = [-0.03, -0.03, -180, -180]
        plot_max = [1.1, 1.1, 180, 180]
        plot_text = [-0.07, -0.07, -180*1.06, -180*1.06]
        for pdx in range(len(plot_quant)):
            if self.fit_for[pdx] == 0:
                continue
            print(plot_quant[pdx])

            plt.figure(figsize=(8, len_lightcurve//2*2.5))
            gs = gridspec.GridSpec(len_lightcurve//2, 2, hspace=0.05,
                                   wspace=0.05)
            for ndx in range(len_lightcurve):
                obj = self.datalist[ndx//2]
                ax = plt.subplot(gs[ndx//2, ndx % 2])
                u = uu[ndx]
                v = vv[ndx]
                u_as_model = np.zeros((len(u), len(wave_model)))
                v_as_model = np.zeros((len(v), len(wave_model)))
                for i in range(0, len(u)):
                    u_as_model[i, :] = u[i]/(wave_model*1.e-6) / rad2as
                    v_as_model[i, :] = v[i]/(wave_model*1.e-6) / rad2as
                magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)

                magu_as = np.copy(obj.spFrequAS)
                magu_as_T3 = np.copy(obj.spFrequAS_T3)
                magu_as_T3_model = np.zeros((4, len(wave_model)))

                if nicer:
                    bl_sort = [2, 3, 5, 0, 4, 1]
                    cl_sort = [0, 3, 2, 1]
                    nchannel = len(magu_as[0])
                    for bl in range(6):
                        magu_as[bl] = (np.linspace(nchannel, 0, nchannel)
                                       + bl_sort[bl]*(nchannel+nchannel//2))
                        magu_as_model[bl] = (np.linspace(nchannel, 0, len(wave_model))
                                             + bl_sort[bl]*(nchannel+nchannel//2))
                    for cl in range(4):
                        magu_as_T3[cl] = (np.linspace(nchannel, 0, nchannel)
                                          + cl_sort[cl]*(nchannel+nchannel//2))
                        magu_as_T3_model[cl] = (np.linspace(nchannel, 0,
                                                            len(wave_model))
                                                + cl_sort[cl]*(nchannel+nchannel//2))
                else:
                    for cl in range(4):
                        magu_as_T3_model[cl] = (obj.max_spf[cl]/(wave_model*1.e-6)
                                                / rad2as)
                val = self.fitdata[pdx*3][ndx]
                err = self.fitdata[pdx*3+1][ndx]
                flag = self.fitdata[pdx*3+2][ndx]
                model = allfitres[ndx][pdx]
                if plot_closure[pdx]:
                    x = magu_as_T3
                    x_model = magu_as_T3_model
                    colors = obj.colors_closure
                    labels = obj.closure_labels
                    prange = 4
                else:
                    x = magu_as
                    x_model = magu_as_model
                    colors = obj.colors_baseline
                    labels = obj.baseline_labels
                    prange = 6
                for i in range(prange):
                    plt.errorbar(x[i, :],
                                 val[i, :]*(1-flag)[i],
                                 err[i, :]*(1-flag)[i],
                                 color=colors[i],
                                 ls='', lw=1, alpha=0.5, capsize=0)
                    plt.scatter(x[i,:],
                                val[i, :]*(1-flag)[i],
                                color=colors[i],
                                alpha=0.5)
                    if nicer and ndx > (len_lightcurve-3):
                        plt.text(x[i, :].mean(), plot_text[pdx],
                                 labels[i],
                                 color=colors[i],
                                 ha='center', va='center')
                    plt.plot(x_model[i, :], model[i, :],
                             color='grey', zorder=100)
                if ndx%2 == 0:
                    plt.ylabel(plot_quant[pdx])
                else:
                    ax.set_yticklabels([])
                plt.ylim(plot_min[pdx], plot_max[pdx])
                if nicer:
                    # ax.set_xticklabels([])
                    ax.set_xticks([])
                else:
                    if ndx > (len_lightcurve-3):
                        plt.xlabel('spatial frequency (1/arcsec)')
                    else:
                        ax.set_xticks([])
            plt.show()

