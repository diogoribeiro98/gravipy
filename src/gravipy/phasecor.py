import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pkg_resources import resource_filename
import pandas as pd
from astropy.io import fits
from scipy import interpolate, optimize, stats
import matplotlib
from astropy.time import Time
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
import os
import glob
from .obs_nights import list_nights
from .gravdata import convert_date, find_nearest, get_angle_header_all, averaging

try:
    from generalFunctions import *
    import dill as pickle
    set_style('show')
except ImportError:
    color1 = '#C02F1D'
    color2 = '#348ABD'
    color3 = '#F26D21'
    color4 = '#7A68A6'


baseline = ['UT43', 'UT42', 'UT41', 'UT32', 'UT31', 'UT21']
colors_baseline = np.array(['k', 'darkblue', color4,
                            color2, 'darkred', color1])
colors_UT = np.array(['k', 'darkblue', color2, color1])


def create_met_files(target='SGRA', create=True, force_new=False,
                     datadir='/data/user/forFrank2/'):
    folders = sorted(glob.glob(datadir + '20??-??-??'))

    s2orbit = np.load(resource_filename('gravipy', 'Datafiles/s2_orbit.npy'))
    s2orbit[:, 1:] *= 1e3
    for folder in folders:
        night = folder[-10:]
        if night in ['2022-06-18']:
            print('Need to fix this night! %s' % night)
            continue
        savedir = folder + '/metrology_files/'
        isExist = os.path.exists(savedir)
        if not isExist:
            print('Creating metrology folder: %s' % savedir)
            os.makedirs(savedir)
        if os.path.isfile('%s%s_%s_angle.npy' % (savedir, target, night)) and not force_new:
            print('%s already exists' % night)
            continue
        else:
            print('Creating metrology files for %s' % night)
        pl_list = sorted(glob.glob(datadir + night
                                   + '/reduced_PL????????'))
        if len(pl_list) < 1:
            raise ValueError('Something wrong with given directory '
                             'No reduction folder in %s'
                             % (datadir + night))
        redfolder = pl_list[-1]

        files = sorted(glob.glob(redfolder + '/*p2vmred.fits'))
        first = True
        tfiles = []
        if target not in ['SGRA', 'S2']:
            for file in files:
                h = fits.open(file)[0].header
                if h['ESO INS SOBJ NAME'] == target:
                    tfiles.append(file)
        else:
            reverse = False
            tfiles_off = []
            for file in files:
                h = fits.open(file)[0].header
                if first:
                    date = convert_date(h['DATE-OBS'])[0]
                    sdx = find_nearest(s2orbit[:,0], date)
                    if (s2orbit[sdx, 0] - date)*300 > 2:
                        raise ValueError('Date offset too big')
                    ra = s2orbit[sdx,1]
                    de = s2orbit[sdx,2]
                    first = False

                if h['ESO FT ROBJ NAME'] != 'IRS16C':
                    continue
                if h['ESO INS SOBJ NAME'] == 'S2':
                    try:
                        if h['ESO INS SOBJ OFFX'] != 0:
                            sobjx = h['ESO INS SOBJ OFFX']
                            sobjy = h['ESO INS SOBJ OFFY']
                            if np.abs(np.abs(sobjx) - np.abs(ra)) < 3 and np.abs(np.abs(sobjy) - np.abs(de)) < 3:
                                tfiles_off.append(file)
                                if np.abs(sobjx - ra) < 3:
                                    reverse = True
                                    print('!!!!!!!!!REVERSE!!!!!!')
                                else:
                                    reverse = False
                        else:
                            tfiles.append(file)
                    except KeyError:
                        continue
            if len(tfiles) == 0 and len(tfiles_off) == 0:
                print('No files available')
                continue
            if reverse and target == 'S2':
                tfiles = tfiles_off
            if not reverse and target == 'SGRA':
                tfiles = tfiles_off
        if len(tfiles) == 0:
            print('No files available')
            continue
        LST = np.array([]).reshape(0)
        OPD_TELFC_CORR = np.array([]).reshape(0, 4, 4)
        OPD_TELFC_MCORR = np.array([]).reshape(0, 4)
        REFANGLE = np.array([]).reshape(0, 4)
        PUPIL = np.array([]).reshape(0, 4, 2)
        FIBER = np.array([]).reshape(0, 4, 2)
        for file in tfiles:
            try:
                m = fits.open(file)['OI_VIS_MET']
            except KeyError:
                print('No met extension in %s' % file)
                continue
            h = fits.open(file)[0].header
            data0 = np.zeros((len(m.data["OPD_TELFC_CORR"])//4))
            data1 = np.zeros((len(m.data["OPD_TELFC_CORR"])//4, 4, 4))
            data2 = np.zeros((len(m.data["OPD_TELFC_MCORR"])//4, 4))
            data3 = np.zeros((len(m.data["OPD_TELFC_MCORR"])//4, 4))
            data4 = np.zeros((len(m.data["OPD_TELFC_MCORR"])//4, 4, 2))
            data5 = np.zeros((len(m.data["OPD_TELFC_MCORR"])//4, 4, 2))
            lst0 = h['LST']/3600
            time = m.data['TIME'][::4]/1e6/3600
            data0 = time + lst0
            data0 = data0 % 24

            data1 = m.data["OPD_TELFC_CORR"].reshape(-1, 4, 4)
            data2 = m.data["OPD_TELFC_MCORR"].reshape(-1, 4)
            data4[:, :, 0] = m.data["PUPIL_U"].reshape(-1, 4)
            data4[:, :, 1] = m.data["PUPIL_V"].reshape(-1, 4)
            data5[:, :, 0] = m.data["FIBER_DU"].reshape(-1, 4)
            data5[:, :, 1] = m.data["FIBER_DV"].reshape(-1, 4)

            for tel in range(4):
                data3[:, tel] = get_angle_header_all(h,  tel, len(data3))
            LST = np.concatenate((LST, data0))
            OPD_TELFC_CORR = np.concatenate((OPD_TELFC_CORR, data1), 0)
            OPD_TELFC_MCORR = np.concatenate((OPD_TELFC_MCORR, data2), 0)
            REFANGLE = np.concatenate((REFANGLE, data3), 0)
            PUPIL = np.concatenate((PUPIL, data4), 0)
            FIBER = np.concatenate((FIBER, data5), 0)

        num_bins = 3600
        angle_bins = np.linspace(0, 360, num_bins)
        bin_TELFC_CORR = np.zeros((num_bins-1, 4, 4))
        bin_TELFC_MCORR = np.zeros((num_bins-1, 4))
        bin_PUPIL = np.zeros((num_bins-1, 4, 2))
        bin_FIBER = np.zeros((num_bins-1, 4, 2))

        for tel in range(4):
            a = REFANGLE[:, tel]
            bin_TELFC_MCORR[:, tel] = stats.binned_statistic(a, OPD_TELFC_MCORR[:, tel],
                                                             bins=angle_bins, statistic="median")[0]
            for dio in range(4):
                bin_TELFC_CORR[:, tel, dio] = stats.binned_statistic(a, OPD_TELFC_CORR[:, tel, dio],
                                                                     bins=angle_bins, statistic="median")[0]
            for dim in range(2):
                bin_PUPIL[:, tel, dim] = stats.binned_statistic(a, PUPIL[:, tel, dim],
                                                                 bins=angle_bins, statistic="median")[0]
                bin_FIBER[:, tel, dim] = stats.binned_statistic(a, FIBER[:, tel, dim],
                                                                bins=angle_bins, statistic="median")[0]
        np.save('%s%s_%s_telfcmcorr.npy' % (savedir, target, night), bin_TELFC_MCORR)
        np.save('%s%s_%s_telfccorr.npy' % (savedir, target, night), bin_TELFC_CORR)
        np.save('%s%s_%s_angle.npy' % (savedir, target, night), angle_bins)
        np.save('%s%s_%s_puil.npy' % (savedir, target, night), bin_PUPIL)
        np.save('%s%s_%s_fiber.npy' % (savedir, target, night), bin_FIBER)


def read_correction(mcor_files, xscale, list_dim=1, fancy=True,
                    wrap=False, lst=False, textpos=15, av=20, std_cut=0.03,
                    met_cut=0.2, bequiet=False):
    """
    Reads in the correction created by 
    TODO add function to create correction
    Input:
    mcor_files:     List of all metrology measurements
    xscale:         X axis of tje mcor_files
    list_dim:       If mcor_files is a list of lists this has to be length
                    of that lists [1]
    fancy:          Use the fancy median correction [True]
    wrap:           Wraps the metrology at 180 degree [False]
    lst:            Has to be true if input is a function of lst [False]
    textpos:        Postition to plot text in figures [15]
    av:             Average for the median of the output figure [20]
    std_cut:        Metrology datapoints are ignored if the std is higher 
                    than this value [0.03]
    met_cut:        Metrology datapoints are ignored if the value is higher 
                    than this [0.2]
    bequiet:        If true all outputs are supressed [False]

    Output:
    x values and array with all corrected metrology measurements
    """
    if list_dim == 1:
        if len(mcor_files) < 10 and fancy:
            print('Very short list of input.')
            print('If you want to use several lists, use list_dim != 1.')
            print('Else not_fancy=True shoule be better...')
        if wrap:
            if lst:
                raise ValueError('Wrap and lst do not work together!')

            def wrap_data(a):
                a1 = a[:a.shape[0]//2]
                a2 = a[a.shape[0]//2+1:]
                b = np.nanmean((a1, a2), 0)
                return b

            TELFC_MCORR_S2 = np.array([wrap_data(np.load(fi)*1e6)
                                       for fi in mcor_files])
            x = np.load(xscale)[:-1]
            x = x[:len(x)//2]
        else:
            TELFC_MCORR_S2 = np.array([np.load(fi)*1e6 for fi in mcor_files])
            x = np.load(xscale)[:-1]

        ndata = len(mcor_files)
        colors = plt.cm.jet(np.linspace(0, 1, ndata))
        len_data = []
        ndata = len(TELFC_MCORR_S2)
        for idx in range(ndata):
            len_data.append(len(TELFC_MCORR_S2[idx, 0, ~np.isnan(TELFC_MCORR_S2[idx, 0])]))
        sort = np.array(len_data).argsort()[::-1]
        TELFC_MCORR_S2 = TELFC_MCORR_S2[sort]

    else:
        if len(mcor_files) != list_dim:
            raise ValueError('list_dim is different then length of input list,'
                             ' input is wrong')
        if wrap:
            raise ValueError('Wrap is not defined yet for multiple input lists')
        else:
            len_lists = [0]
            for ldx in range(list_dim):
                in_mcorr = np.array([np.load(fi)*1e6 for fi in mcor_files[ldx]])
                len_lists.append(len(in_mcorr)+len_lists[-1])
                len_data = []
                ndata = len(in_mcorr)
                for idx in range(ndata):
                    len_data.append(len(in_mcorr[idx, 0, ~np.isnan(in_mcorr[idx,0])]))
                sort = np.array(len_data).argsort()[::-1]
                if ldx == 0:
                    TELFC_MCORR_S2 = in_mcorr[sort]
                else:
                    TELFC_MCORR_S2 = np.concatenate((TELFC_MCORR_S2,
                                                     in_mcorr[sort]), 0)
            x = np.load(xscale)[:-1]
            ndata = len(TELFC_MCORR_S2)
            colors = plt.cm.jet(np.linspace(0, 1, ndata))
            colors_lists = ['k', color2, color3]

    if not bequiet:
        gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.1)
        plt.figure(figsize=(8, 6))
        for tel in range(4):
            ax = plt.subplot(gs[tel % 2, tel // 2])
            for idx in range(ndata):
                data = (TELFC_MCORR_S2[idx, :, tel]
                        - np.nanmean(TELFC_MCORR_S2[idx, :, tel]))
                if list_dim == 1:
                    plt.plot(x, data,
                             marker='.', ls='', alpha=0.5, color=colors[idx])
                else:
                    ndx = np.where(np.array(len_lists) <= idx)[0][-1]
                    plt.plot(x, data, marker='.', ls='', 
                             alpha=0.2, color=colors_lists[ndx])
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3,0.3)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel % 2 != 1:
                ax.set_xticklabels([])
            else:
                if lst:
                    plt.xlabel('LST [h]')
                else:
                    plt.xlabel('Ref. angle [deg]')
        plt.show()

    if not fancy:
        TELFC_MCORR_S2_corr = np.copy(TELFC_MCORR_S2)
        for idx in range(ndata):
            for tel in range(4):
                TELFC_MCORR_S2_corr[idx, :, tel] -= np.nanmean(TELFC_MCORR_S2_corr[idx, :, tel])

        for tel in range(4):
            TELFC_MCORR_S2_corr[:, :, tel] -= np.nanmean(TELFC_MCORR_S2_corr[:, :, tel])

        gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.1)
        plt.figure(figsize=(8, 6))
        for tel in range(4):
            ax = plt.subplot(gs[tel % 2, tel // 2])
            for idx in range(ndata):
                data = TELFC_MCORR_S2_corr[idx, :, tel]
                if list_dim == 1:
                    plt.plot(x, data, marker='.', ls='', alpha=0.5, color='k')
                else:
                    ndx = np.where(np.array(len_lists) <= idx)[0][-1]
                    plt.plot(x, data, marker='.', ls='',
                             alpha=0.1, color=colors_lists[ndx])
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3,0.3)
            plt.plot(averaging(x, av),
                     averaging(np.nanmedian(TELFC_MCORR_S2_corr[:, : ,tel], 0), av),
                     color=color1)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel % 2 != 1:
                ax.set_xticklabels([])
            else:
                if lst:
                    plt.xlabel('LST [h]')
                else:
                    plt.xlabel('Ref. angle [deg]')
        plt.show()

        return x, TELFC_MCORR_S2_corr

    TELFC_MCORR_S2_corr = np.copy(TELFC_MCORR_S2)
    TELFC_MCORR_S2_av = np.zeros((ndata, TELFC_MCORR_S2.shape[1]//av, 4))
    TELFC_MCORR_S2_std = np.zeros((ndata, TELFC_MCORR_S2.shape[1]//av, 4))
    for idx in range(ndata):
        for tel in range(4):
            TELFC_MCORR_S2_av[idx, :, tel] = averaging(TELFC_MCORR_S2[idx, :, tel], av)[:-1]
            TELFC_MCORR_S2_std[idx, :, tel] = averaging_std(TELFC_MCORR_S2[idx, :, tel], av)[:-1]
    for tel in range(4):
        TELFC_MCORR_S2_corr[0, :, tel] -= np.nanmean(TELFC_MCORR_S2_corr[0, :, tel])
        TELFC_MCORR_S2_av[0, :, tel] -= np.nanmean(TELFC_MCORR_S2_av[0, :, tel])
        TELFC_MCORR_S2_corr[0, np.where(np.abs(TELFC_MCORR_S2_corr[0, :, tel]) > met_cut),tel] *= np.nan
        TELFC_MCORR_S2_av[0, np.where(np.abs(TELFC_MCORR_S2_av[0, :, tel]) > met_cut),tel] *= np.nan

    for idx in range(1, ndata):
        if idx == 1:
            corr = TELFC_MCORR_S2_av[0]
        else:
            corr = np.nanmedian(TELFC_MCORR_S2_av[:(idx-1)],0)

        for tel in range(4):
            TELFC_MCORR_S2_av[idx, [np.where(TELFC_MCORR_S2_std[idx, :, tel] > std_cut)], tel] = np.nan
            mask = ~np.isnan(corr[:, tel])*~np.isnan(TELFC_MCORR_S2_av[idx, :, tel])
            if len(mask[mask]) == 0:
                TELFC_MCORR_S2_av[idx, :, tel] -= np.nanmedian(TELFC_MCORR_S2_av[idx, :, tel])
                TELFC_MCORR_S2_corr[idx, :, tel] -= np.nanmedian(TELFC_MCORR_S2_corr[idx, :, tel])
                if not bequiet:
                    if tel == 0:
                        print('%i no overlap' % idx)
            else:
                mdiff = np.mean(TELFC_MCORR_S2_av[idx, mask, tel])-np.mean(corr[mask, tel])
                TELFC_MCORR_S2_corr[idx, :, tel] -= mdiff
                TELFC_MCORR_S2_av[idx, :, tel] -= mdiff
                if not bequiet:
                    if tel == 0:
                        print(idx, mdiff)
            if np.nanstd(TELFC_MCORR_S2_corr[idx, :, tel]) > 0.1:
                if not bequiet:
                    print('%i, %i: std of data too big' % (idx, tel))
                TELFC_MCORR_S2_corr[idx, :, tel] *= np.nan
                TELFC_MCORR_S2_av[idx, :, tel] *= np.nan

            TELFC_MCORR_S2_corr[idx, np.where(np.abs(TELFC_MCORR_S2_corr[idx, : ,tel]) > met_cut), tel] *= np.nan
            TELFC_MCORR_S2_av[idx, np.where(np.abs(TELFC_MCORR_S2_av[idx, : ,tel]) > met_cut), tel] *= np.nan

    if not bequiet:
        gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.1)
        plt.figure(figsize=(8, 6))
        for tel in range(4):
            ax = plt.subplot(gs[tel%2, tel//2])
            for idx in range(ndata):
                data = TELFC_MCORR_S2_corr[idx, :, tel]
                plt.plot(x, data,
                         marker='.', ls='', alpha=0.5, color=colors[idx])
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3, 0.3)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel % 2 != 1:
                ax.set_xticklabels([])
            else:
                if lst:
                    plt.xlabel('LST [h]')
                else:
                    plt.xlabel('Ref. angle [deg]')
        plt.show()

    for tel in range(4):
        TELFC_MCORR_S2_corr[:, :, tel] -= np.nanmean(TELFC_MCORR_S2_corr[:, :, tel])

    if not bequiet:
        gs = gridspec.GridSpec(2, 2, wspace=0.1, hspace=0.1)
        plt.figure(figsize=(8, 6))
        for tel in range(4):
            ax = plt.subplot(gs[tel % 2, tel//2])
            for idx in range(ndata):
                data = TELFC_MCORR_S2_corr[idx, :, tel]
                if list_dim == 1:
                    plt.plot(x, data, marker='.', ls='', alpha=0.5, color='k')
                else:
                    ndx = np.where(np.array(len_lists) <= idx)[0][-1]
                    plt.plot(x, data, marker='.', ls='',
                             alpha=0.1, color=colors_lists[ndx])
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3,0.3)
            plt.plot(averaging(x, av),
                     averaging(np.nanmedian(TELFC_MCORR_S2_corr[:, :, tel], 0),
                               av),
                     color=color1)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel % 2 != 1:
                ax.set_xticklabels([])
            else:
                if lst:
                    plt.xlabel('LST [h]')
                else:
                    plt.xlabel('Ref. angle [deg]')
        plt.show()

    return x, TELFC_MCORR_S2_corr


class GravPhaseNight():
    def __init__(self, night=None, verbose=True,
                 reddir=None, datadir='/data/user/forFrank2/',
                 onlysgra=False, calibrator=None,
                 s2_offx=None, ignore_files=[],
                 usepandas=False, pandasfile=None,
                 full_folder=False):
        """
        Package to do the full phase calibration, poscor, and
        metrology correction

        To load the night data several options are available:
        night       : night which shall be used
        verbose     : show printouts [True]
        reddir      : allows to pick a specific reduction directory [None]
        datadir     : pick directory with data [/data/user/forFrank2/]
        onlysgra    : picks only sgra files for science [False]
        calibrator  : the calibrator to use, if None use defaults [None]
        s2_offx     : SOBJ.OFFX of S2 files, if None use defaults [None]
        ignore_files: List of files to be ignored
        usepandas   : if True gets flux values for 2019 data [False]
        pandasfile  : allows to pick different file if not None [None]
        full_folder : Just load everything (for testing) [False]

        Main functionality is in function:
        process_night
        """
        self.night = night
        self.verbose = verbose

        nights = []
        calibrators = []
        offsets = []
        for _n in list_nights:
            nights.append(_n['night'])
            calibrators.append(_n['calibrator'])
            offsets.append(_n['s2off'])
        if night is None:
            print(nights)
            raise ValueError('Night has to be given as argument')

        if full_folder:
            usepandas = True
            self.folder = night
            if calibrator is None:
                raise ValueError('For full_folder you need to give a calibrator')
            self.calibrator = calibrator

        else:
            try:
                if calibrator is None:
                    if self.verbose:
                        print("using default calibrator")
                    self.calibrator = calibrators[nights.index(night)]
                else:
                    if self.verbose:
                        print("using custom calibrator")
                    self.calibrator = calibrator
                if self.verbose:
                    print('Night:      %s \nCalibrator: %s' % (night,
                                                               self.calibrator))
            except ValueError:
                if self.verbose:
                    print('Night is not available, try one of those:')
                    print(nights)
                raise ValueError('Night is not available')
            if reddir is None:
                pl_list = sorted(glob.glob(datadir + night
                                           + '/reduced_PL????????'))
                if len(pl_list) < 1:
                    raise ValueError('Something wrong with given directory '
                                     'No reduction folder in %s'
                                     % (datadir + night))
                self.folder = pl_list[-1]
            else:
                self.folder = datadir + night + '/' + reddir
        if self.verbose:
            print('Data from:  %s' % self.folder)
        self.bl_array = np.array([[0, 1],
                                  [0, 2],
                                  [0, 3],
                                  [1, 2],
                                  [1, 3],
                                  [2, 3]])

        allfiles = sorted(glob.glob(self.folder + '/GRAVI*dualscivis.fits'))
        if len(allfiles) == 0:
            raise ValueError('No files found, most likely something is wrong'
                             ' with the given reduction folder')

        if s2_offx is None:
            s2_offx = offsets[nights.index(night)]
        self.s2_offx = s2_offx
        s2data = np.load(resource_filename('gravipy', 'Datafiles/s2_orbit.npy'))
        if full_folder:
            sg_files = allfiles
            s2_files = allfiles
        else:
            sg_files = []
            s2_files = []
            for file in allfiles:
                h = fits.open(file)[0].header
                if h['ESO FT ROBJ NAME'] != 'IRS16C':
                    continue
                if h['ESO INS SOBJ NAME'] == 'S2':
                    if h['ESO INS SOBJ OFFX'] == s2_offx:
                        if file not in ignore_files:
                            s2_files.append(file)
                    else:
                        d = convert_date(h['DATE-OBS'])[0]
                        _x, _y = -s2data[find_nearest(s2data[:, 0], d)][1:]*1e3
                        sobjx = h['ESO INS SOBJ X']
                        sobjy = h['ESO INS SOBJ Y']
                        sobjoffx = h['ESO INS SOBJ OFFX']
                        sobjoffy = h['ESO INS SOBJ OFFY']
                        if onlysgra:
                            if s2_offx != 0.0:
                                if sobjoffx != 0.0 or sobjoffy != 0.0:
                                    if self.verbose:
                                        print('File with separation (%i,%i) not '
                                              'an SGRA file, will be ignored'
                                              % (sobjx, sobjy))
                                    continue
                            else:
                                if np.abs(sobjoffx - _x) > 10:
                                    if self.verbose:
                                        print('File with separation (%i,%i) not on S2 '
                                              'orbit, will be ignored' % (sobjx, sobjy))
                                    continue
                                if np.abs(sobjoffy - _y) > 10:
                                    if self.verbose:
                                        print('File with separation (%i,%i) not on S2 '
                                              'orbit, will be ignored' % (sobjx, sobjy))
                                    continue
                        if file not in ignore_files:
                            sg_files.append(file)
        if self.verbose:
            print('            %i SGRA files \n            %i S2 files' 
                  % (len(sg_files), len(s2_files)))
        self.s2_files = s2_files
        self.sg_files = sg_files
        self.ndit = len(fits.open(self.s2_files[0])['OI_VIS', 11].data['TIME'])//6
        if self.verbose:
            print('NDIT:       %i' % self.ndit)
        try:
            year = int(night[:4])
            if year > 2019 and usepandas:
                if self.verbose:
                    print('No flux data in pandas for %i' % year)
                usepandas = False
        except ValueError:
            pass

        if usepandas:
            ################
            # read in flux from pandas
            ################
            if pandasfile is not None:
                pand = pandasfile
                if self.verbose:
                    print('Use given pandas')
            else:
                if self.verbose:
                    print('Read in pandas')
                pandasfile = resource_filename('gravipy',
                                               'Datafiles/GRAVITY_2019data.object')
                pand = pd.read_pickle(pandasfile)

            sg_flux = []
            sg_flux_p1 = []
            sg_flux_p2 = []
            sg_header = []
            s2_pos = []
            for fdx, file in enumerate(sg_files):
                d = fits.open(file)
                h = d[0].header
                sg_header.append(h)
                obsdate = h['DATE-OBS']
                p1 = np.mean(pand["flux p1 [%S2]"].loc[pand['DATE-OBS'] == obsdate])
                p2 = np.mean(pand["flux p2 [%S2]"].loc[pand['DATE-OBS'] == obsdate])
                sg_fr = (p1+p2)/2
                if np.isnan(sg_fr):
                    print('%s has no flux value' % file)
                sg_flux.append(sg_fr)
                sg_flux_p1.append(p1)
                sg_flux_p2.append(p2)

            self.sg_flux = sg_flux
            self.sg_flux_p1 = sg_flux_p1
            self.sg_flux_p2 = sg_flux_p2
            self.sg_header = sg_header
        if self.verbose:
            print('\n\n')

    def get_corrections(self, bequiet=False):
        folder = resource_filename('gravipy', 'met_corrections/')
        corr_ang = sorted(glob.glob(folder + 'correction*'))
        corr_lst = sorted(glob.glob(folder + 'lst_correction*'))
        corr_bl = sorted(glob.glob(folder + 'bl_correction*'))

        corrections_dict = {} 
        for cor in corr_ang:
            name = cor[cor.find('/correction_')+12:-4]
            data = np.load(cor)
            if len(data) == 5:
                interp = []
                x = data[0]
                y = data[1:]
                for tel in range(4):
                    interp.append(interpolate.interp1d(x, y[tel]))
                corrections_dict[name] = interp
            else:
                raise ValueError('Weird dimension of npy file')

        for cor in corr_lst:
            name = 'lst_' + cor[cor.find('/lst_correction_')+16:-4]
            data = np.load(cor)
            if len(data) == 5:
                interp = []
                x = data[0]
                y = data[1:]
                for tel in range(4):
                    interp.append(interpolate.interp1d(x, y[tel]))
                corrections_dict[name] = interp
            else:
                raise ValueError('Weird dimension of npy file')

        for cor in corr_bl:
            name = 'bl_' + cor[cor.find('/bl_correction_')+15:-4]
            data = np.load(cor)
            if len(data) == 7:
                interp = []
                x = data[0]
                y = data[1:]
                for bl in range(6):
                    interp.append(interpolate.interp1d(x, y[bl]))
                corrections_dict[name] = interp
            else:
                raise ValueError('Weird dimension of npy file')

        if not bequiet:
            print('Available datasets:')
            print(corrections_dict.keys())
        return corrections_dict

    def get_phasecorrections(self, bequiet=False):
        folder = resource_filename('gravipy', 'met_corrections/')
        corr_ang = sorted(glob.glob(folder + 'phasecorrection*'))
        corrections_dict = {} 
        for cor in corr_ang:
            name = cor[cor.find('/phasecorrection_')+17:-4]
            data = np.load(cor)
            if len(data) == 7:
                interp = []
                x = data[0]
                y = data[1:]
                for tel in range(6):
                    interp.append(interpolate.interp1d(x, y[tel]))
                corrections_dict[name] = interp
            else:
                raise ValueError('Weird dimension of npy file')
        if not bequiet:
            print('Available datasets:')
            print(corrections_dict.keys())
        return corrections_dict

    def process_night(self, mode,  fluxcut=0.0, subspacing=1,
                      poscor=True, plot=False,
                      fitstart=0, fitstop=-1, linear_cor=False,
                      save=False, ret=False, phasecorrection=None):
        """
        Function to corect, calibrate & poscor data
        mode        : mode for the correction. To see whats available run:
                      get_corrections()
                      if None, does not use any correction
        fluxcut     : if > 0 only uses the data which is brighter [0]
        subspacing  : spaces the applied corrections more than the data [1]
        poscor      : apply poscor [True]
        plot        : Show some plots [False]
        linear_cor  : subtracts a polynom of 1st degree from each
                      baseline [False]    
        fitstart    : file number from which to start the linear fit [0]
        fitstop     : file number where to stop measure the linear fit [-1]
        save        : saves the final data under one of the following 
                      folders [False]
                      (depending on the options)
                      /data/user/forFrank2/ + night + /reduced_PL20200513/correction_ + mode + _new/
                      /data/user/forFrank2/ + night + /reduced_PL20200513_1frame/correction_ + mode + _new/ 
                      if save is a string, save files will be save to string value
        ret         : Returns list with all results
        """
        ndit = self.ndit

        s2_files = self.s2_files
        sg_files = self.sg_files
        if mode is None:
            correction = False
            if self.verbose:
                print('No correction for metrology systematics used')
        else:
            correction = True
            if isinstance(mode, str):
                corrections_dict = self.get_corrections(bequiet=True)
                try:
                    interp_list = corrections_dict[mode]
                except KeyError:
                    print('mode not avialable, use one of those:')
                    print(corrections_dict.keys())
                    print('More corrections in: '
                          '/data/user/fwidmann/Phase_fit_cor/corrections_data/')
                    raise KeyError

                if 'lst' in mode:
                    lst_corr = True
                    if ndit == 1:
                        raise ValueError('Ndit 1 and lst correction is not '
                                         'implemented properly')
                    print('Apply LST correction')
                else:
                    lst_corr = False

                if 'bl' in mode:
                    bl_corr = True
                    print('Apply BL correction')
                else:
                    bl_corr = False
            else:
                if len(mode) == 4:
                    bl_corr = False
                    lst_corr = False
                    interp_list = mode
                    if self.verbose:
                        print('using the interpolations given in input')
                else:
                    raise ValueError('If interpolations are given mode has '
                                     'to be a list of four')

        if phasecorrection is not None:
            cor_cor = True
            corrections_dict = self.get_phasecorrections(bequiet=True)
            try:
                interp_list_phase = corrections_dict[phasecorrection]
            except KeyError:
                print('mode not avialable, use one of those:')
                print(corrections_dict.keys())
                raise KeyError
        else:
            cor_cor = False

        try:
            d = fits.open(sg_files[0])
        except IndexError:
            d = fits.open(s2_files[0])
        wave = d['OI_WAVELENGTH', 11].data['EFF_WAVE']
        self.wave = wave
        nchannel = len(wave)
        sg_visphi_p1 = np.zeros((len(sg_files), ndit*6, nchannel))*np.nan
        sg_visphi_p2 = np.zeros((len(sg_files), ndit*6, nchannel))*np.nan
        sg_visphi_err_p1 = np.zeros((len(sg_files), ndit*6, nchannel))*np.nan
        sg_visphi_err_p2 = np.zeros((len(sg_files), ndit*6, nchannel))*np.nan
        sg_t = np.zeros((len(sg_files), ndit))*np.nan
        sg_lst = np.zeros((len(sg_files), ndit))*np.nan
        sg_ang = np.zeros((len(sg_files), ndit))*np.nan
        sg_u = np.zeros((len(sg_files), ndit*6, nchannel))*np.nan
        sg_v = np.zeros((len(sg_files), ndit*6, nchannel))*np.nan
        sg_u_raw = np.zeros((len(sg_files), ndit*6))*np.nan
        sg_v_raw = np.zeros((len(sg_files), ndit*6))*np.nan
        sg_correction = np.zeros((len(sg_files), ndit*6, nchannel))*np.nan
        sg_correction_wo = np.zeros((len(sg_files), ndit*6, nchannel))*np.nan

        for fdx, file in enumerate(sg_files):
            d = fits.open(file)
            h = d[0].header
            # if fluxcut > 0:
            #     if f < self.sg_flux[fdx]:
            #         continue
            sg_t[fdx] = d['OI_VIS', 12].data['MJD'][::6]
            o_ndit = h['ESO DET2 NDIT']
            lst0 = h['LST']/3600
            time = d['OI_VIS', 11].data['TIME'][::6]/3600
            time -= time[0]
            sg_lst[fdx] = time + lst0

            U = d['OI_VIS', 11].data['UCOORD']
            V = d['OI_VIS', 11].data['VCOORD']
            sg_u_raw[fdx] = U
            sg_v_raw[fdx] = V
            wave = d['OI_WAVELENGTH', 11].data['EFF_WAVE']
            U_as = np.zeros((ndit*6, nchannel))
            V_as = np.zeros((ndit*6, nchannel))
            for bl in range(ndit*6):
                for wdx, wl in enumerate(wave):
                    U_as[bl, wdx] = U[bl]/wl * np.pi / 180. / 3600./1000
                    V_as[bl, wdx] = V[bl]/wl * np.pi / 180. / 3600./1000
            sg_u[fdx] = U_as
            sg_v[fdx] = V_as

            angle = [get_angle_header_all(h, i, o_ndit) for i in range(4)]
            sg_ang[fdx] = np.mean([get_angle_header_all(h, i, ndit)
                                   for i in range(4)], 0)
            wave = d['OI_WAVELENGTH', 12].data['EFF_WAVE']*1e6
            dlambda = d['OI_WAVELENGTH', 11].data['EFF_BAND']/2*1e6
            fullcor = np.zeros((6*ndit, len(wave)))

            if correction:
                try:
                    for base in range(6):
                        if bl_corr:
                            if lst_corr:
                                if subspacing != 1:
                                    lstdif = sg_lst[fdx][-1] - sg_lst[fdx][-2]
                                    lst_s = np.linspace(sg_lst[fdx][0], sg_lst[fdx][-1] + lstdif, len(sg_lst[fdx])*subspacing)
                                    cor = -averaging(interp_list[base](lst_s), subspacing)[:-1]
                                else:
                                    cor = -interp_list[base](sg_lst[fdx])
                            else:
                                t1 = self.bl_array[base][0]
                                t2 = self.bl_array[base][1]
                                ang1 = angle[t1]
                                ang2 = angle[t2]
                                mang = (ang1 + ang2)/2
                                if subspacing != 1:
                                    mangdif = (mang[-1]-mang[-2])/2
                                    mang_s = np.linspace(mang[0] - mangdif, mang[-1] + mangdif, len(mang)*subspacing)
                                    cor = -averaging(interp_list[base](mang_s), subspacing)[:-1]
                                else:
                                    cor = -interp_list[base](mang)

                        else:
                            t1 = self.bl_array[base][0]
                            t2 = self.bl_array[base][1]
                            if lst_corr:
                                if subspacing != 1:
                                    lstdif = sg_lst[fdx][-1] - sg_lst[fdx][-2]
                                    lst_s = np.linspace(sg_lst[fdx][0], sg_lst[fdx][-1] + lstdif, len(sg_lst[fdx])*subspacing)
                                    cor1 = averaging(interp_list[t1](lst_s), subspacing)[:-1]
                                    cor2 = averaging(interp_list[t2](lst_s), subspacing)[:-1]
                                else:
                                    cor1 = interp_list[t1](sg_lst[fdx])
                                    cor2 = interp_list[t2](sg_lst[fdx])  
                            else:
                                ang1 = angle[t1]
                                ang2 = angle[t2]
                                if subspacing != 1:
                                    angdif1 = (ang1[-1]-ang1[-2])/2
                                    angdif2 = (ang2[-1]-ang2[-2])/2
                                    ang1_s = np.linspace(ang1[0] - angdif1, ang1[-1] + angdif1, len(ang1)*subspacing)
                                    ang2_s = np.linspace(ang2[0] - angdif2, ang2[-1] + angdif2, len(ang2)*subspacing)
                                    cor1 = averaging(interp_list[t1](ang1_s), subspacing)[:-1]
                                    cor2 = averaging(interp_list[t2](ang2_s), subspacing)[:-1]
                                else:
                                    cor1 = interp_list[t1](ang1)
                                    cor2 = interp_list[t2](ang2)

                            cor = cor1-cor2
                            if mode in ['lst_standard', 'standard']:
                                cor *= 1e6
                        if ndit == 1:
                            cor = np.mean(cor)

                        wcor = np.zeros((ndit, len(wave)))
                        for w in range(len(wave)):
                            wcor[:, w] = cor/wave[w]*360
                        fullcor[base::6] = wcor
                    sg_correction_wo[fdx] = fullcor

                    if cor_cor:
                        for base in range(6):
                            t1 = self.bl_array[base][0]
                            t2 = self.bl_array[base][1]
                            ang1 = angle[t1]
                            ang2 = angle[t2]
                            mang = (ang1 + ang2)/2
                            if subspacing != 1:
                                mangdif = (mang[-1]-mang[-2])/2
                                mang_s = np.linspace(mang[0] - mangdif, mang[-1] + mangdif, len(mang)*subspacing)
                                phasecor = averaging(interp_list_phase[base](mang_s), subspacing)[:-1]
                            else:
                                phasecor = interp_list_phase[base](mang)
                            if ndit == 1:
                                phasecor = np.mean(phasecor)
                            wphasecor = np.zeros((ndit, len(wave)))
                            for w in range(len(wave)):
                                wphasecor[:,w] = phasecor/wave[w]*360
                            fullcor[base::6] -= wphasecor
                    fullcor[np.isnan(fullcor)] = 0

                except ValueError:
                    fullcor = 0

                sg_visphi_p1[fdx] = d['OI_VIS', 11].data['VISPHI'] + fullcor
                sg_visphi_p2[fdx] = d['OI_VIS', 12].data['VISPHI'] + fullcor
            else:
                sg_visphi_p1[fdx] = d['OI_VIS', 11].data['VISPHI']
                sg_visphi_p2[fdx] = d['OI_VIS', 12].data['VISPHI']

            sg_correction[fdx] = fullcor

            flag1 = d['OI_VIS', 11].data['FLAG']
            flag2 = d['OI_VIS', 12].data['FLAG']

            sg_visphi_err_p1[fdx] = d['OI_VIS', 11].data['VISPHIERR']
            sg_visphi_err_p2[fdx] = d['OI_VIS', 12].data['VISPHIERR']

            sg_visphi_p1[fdx][np.where(flag1 == True)] = np.nan
            sg_visphi_p2[fdx][np.where(flag2 == True)] = np.nan
            sg_visphi_err_p1[fdx][np.where(flag1 == True)] = np.nan
            sg_visphi_err_p2[fdx][np.where(flag2 == True)] = np.nan

        if len(sg_files) > 0:
            self.sg_u_raw = sg_u_raw
            self.sg_v_raw = sg_v_raw
            self.wave = wave
            self.dlambda = dlambda

        s2_visphi_p1 = np.zeros((len(s2_files), ndit*6, nchannel))
        s2_visphi_p2 = np.zeros((len(s2_files), ndit*6, nchannel))
        s2_visphi_err_p1 = np.zeros((len(s2_files), ndit*6, nchannel))
        s2_visphi_err_p2 = np.zeros((len(s2_files), ndit*6, nchannel))
        s2_t = np.zeros((len(s2_files), ndit))
        s2_lst = np.zeros((len(s2_files), ndit))
        s2_ang = np.zeros((len(s2_files), ndit))
        s2_u = np.zeros((len(s2_files), ndit*6, nchannel))
        s2_v = np.zeros((len(s2_files), ndit*6, nchannel))
        s2_u_raw = np.zeros((len(s2_files), ndit*6))
        s2_v_raw = np.zeros((len(s2_files), ndit*6))
        s2_correction = np.zeros((len(s2_files), ndit*6, nchannel))*np.nan
        s2_correction_wo = np.zeros((len(s2_files), ndit*6, nchannel))*np.nan

        for fdx, file in enumerate(s2_files):
            d = fits.open(file)
            h = d[0].header
            s2_t[fdx] = d['OI_VIS', 12].data['MJD'][::6]

            o_ndit = h['ESO DET2 NDIT']
            lst0 = h['LST']/3600
            time = d['OI_VIS', 11].data['TIME'][::6]/3600
            time -= time[0]
            s2_lst[fdx] = time + lst0

            U = d['OI_VIS', 11].data['UCOORD']
            V = d['OI_VIS', 11].data['VCOORD']
            s2_u_raw[fdx] = U
            s2_v_raw[fdx] = V
            wave = d['OI_WAVELENGTH', 11].data['EFF_WAVE']
            U_as = np.zeros((ndit*6, nchannel))
            V_as = np.zeros((ndit*6, nchannel))
            for bl in range(ndit*6):
                for wdx, wl in enumerate(wave):
                    U_as[bl, wdx] = U[bl]/wl * np.pi / 180. / 3600./1000
                    V_as[bl, wdx] = V[bl]/wl * np.pi / 180. / 3600./1000
            s2_u[fdx] = U_as
            s2_v[fdx] = V_as

            angle = [get_angle_header_all(h, i, o_ndit) for i in range(4)]
            s2_ang[fdx] = np.mean([get_angle_header_all(h, i, ndit) for i in range(4)], 0)
            wave = d['OI_WAVELENGTH', 12].data['EFF_WAVE']*1e6
            fullcor = np.zeros((6*ndit, len(wave)))

            if correction:
                try:
                    for base in range(6):
                        if bl_corr:
                            if lst_corr:
                                if subspacing != 1:
                                    lstdif = s2_lst[fdx][-1] - s2_lst[fdx][-2]
                                    lst_s = np.linspace(s2_lst[fdx][0], s2_lst[fdx][-1] + lstdif, len(s2_lst[fdx])*subspacing)
                                    cor = -averaging(interp_list[base](lst_s), subspacing)[:-1]
                                else:
                                    cor = -interp_list[base](s2_lst[fdx])
                            else:
                                t1 = self.bl_array[base][0]
                                t2 = self.bl_array[base][1]
                                ang1 = angle[t1]
                                ang2 = angle[t2]
                                mang = (ang1 + ang2)/2
                                if subspacing != 1:
                                    mangdif = (mang[-1]-mang[-2])/2
                                    mang_s = np.linspace(mang[0] - mangdif, mang[-1] + mangdif, len(mang)*subspacing)
                                    cor = -averaging(interp_list[base](mang_s), subspacing)[:-1]
                                else:
                                    cor = -interp_list[base](mang)

                        else:
                            t1 = self.bl_array[base][0]
                            t2 = self.bl_array[base][1]
                            if lst_corr:
                                if subspacing != 1:
                                    lstdif = s2_lst[fdx][-1] - s2_lst[fdx][-2]
                                    lst_s = np.linspace(s2_lst[fdx][0], s2_lst[fdx][-1] + lstdif, len(s2_lst[fdx])*subspacing)
                                    cor1 = averaging(interp_list[t1](lst_s), subspacing)[:-1]
                                    cor2 = averaging(interp_list[t2](lst_s), subspacing)[:-1]
                                else:
                                    cor1 = interp_list[t1](s2_lst[fdx])
                                    cor2 = interp_list[t2](s2_lst[fdx])  
                            else:
                                ang1 = angle[t1]
                                ang2 = angle[t2]
                                if subspacing != 1:
                                    angdif1 = (ang1[-1]-ang1[-2])/2
                                    angdif2 = (ang2[-1]-ang2[-2])/2
                                    ang1_s = np.linspace(ang1[0] - angdif1, ang1[-1] + angdif1, len(ang1)*subspacing)
                                    ang2_s = np.linspace(ang2[0] - angdif2, ang2[-1] + angdif2, len(ang2)*subspacing)
                                    cor1 = averaging(interp_list[t1](ang1_s), subspacing)[:-1]
                                    cor2 = averaging(interp_list[t2](ang2_s), subspacing)[:-1]
                                else:
                                    cor1 = interp_list[t1](ang1)
                                    cor2 = interp_list[t2](ang2)

                            cor = cor1-cor2
                            if mode in ['lst_standard', 'standard']:
                                cor *= 1e6

                        if ndit == 1:
                            cor = np.mean(cor)

                        wcor = np.zeros((ndit, len(wave)))
                        for w in range(len(wave)):
                            wcor[:, w] = cor/wave[w]*360
                        fullcor[base::6] = wcor
                    s2_correction_wo[fdx] = fullcor

                    if cor_cor:
                        for base in range(6):
                            t1 = self.bl_array[base][0]
                            t2 = self.bl_array[base][1]
                            ang1 = angle[t1]
                            ang2 = angle[t2]
                            mang = (ang1 + ang2)/2
                            if subspacing != 1:
                                mangdif = (mang[-1]-mang[-2])/2
                                mang_s = np.linspace(mang[0] - mangdif, mang[-1] + mangdif, len(mang)*subspacing)
                                phasecor = averaging(interp_list_phase[base](mang_s), subspacing)[:-1]
                            else:
                                phasecor = interp_list_phase[base](mang)
                            if ndit == 1:
                                phasecor = np.mean(phasecor)
                            wphasecor = np.zeros((ndit, len(wave)))
                            for w in range(len(wave)):
                                wphasecor[:,w] = phasecor/wave[w]*360
                            fullcor[base::6] -= wphasecor

                    fullcor[np.isnan(fullcor)] = 0
                except ValueError:
                    fullcor = 0
                s2_visphi_p1[fdx] = d['OI_VIS', 11].data['VISPHI'] + fullcor
                s2_visphi_p2[fdx] = d['OI_VIS', 12].data['VISPHI'] + fullcor
            else:
                s2_visphi_p1[fdx] = d['OI_VIS', 11].data['VISPHI']
                s2_visphi_p2[fdx] = d['OI_VIS', 12].data['VISPHI']

            s2_correction[fdx] = fullcor

            flag1 = d['OI_VIS', 11].data['FLAG']
            flag2 = d['OI_VIS', 12].data['FLAG']

            s2_visphi_err_p1[fdx] = d['OI_VIS', 11].data['VISPHIERR']
            s2_visphi_err_p2[fdx] = d['OI_VIS', 12].data['VISPHIERR']

            s2_visphi_p1[fdx][np.where(flag1 ==True)] = np.nan
            s2_visphi_p2[fdx][np.where(flag2 ==True)] = np.nan
            s2_visphi_err_p1[fdx][np.where(flag1 ==True)] = np.nan
            s2_visphi_err_p2[fdx][np.where(flag2 ==True)] = np.nan

        self.s2_u_raw = s2_u_raw
        self.s2_v_raw = s2_u_raw

        try:
            tstart = np.nanmin((np.nanmin(s2_t), np.nanmin(sg_t)))
        except ValueError:
            tstart = np.nanmin(s2_t)
        s2_t = (s2_t-tstart)*24*60
        sg_t = (sg_t-tstart)*24*60

        ##########################
        # Calibrate
        ##########################

        calib_file = [i for i, s in enumerate(s2_files) if self.calibrator in s]
        if len(calib_file) != 1:
            print(calib_file)
            raise ValueError('Something wrong with calib file')
        calib_file = calib_file[0]
        c_p1 = s2_visphi_p1[calib_file]
        c_p2 = s2_visphi_p2[calib_file]

        mc_p1 = np.zeros((ndit, 6, nchannel))
        mc_p2 = np.zeros((ndit, 6, nchannel))
        cal_p1 = np.zeros_like(c_p1, dtype=np.complex)
        cal_p2 = np.zeros_like(c_p2, dtype=np.complex)

        for bl in range(6):
            mc_p1[:, bl, :] = c_p1[bl::6, :]
            mc_p2[:, bl, :] = c_p2[bl::6, :]

        mc_p1 = np.exp(1j*mc_p1/180*np.pi)
        mc_p1 = np.nanmean(mc_p1, 0)
        mc_p2 = np.exp(1j*mc_p2/180*np.pi)
        mc_p2 = np.nanmean(mc_p2, 0)

        for dit in range(ndit):
            cal_p1[dit*6:(dit+1)*6] = mc_p1
            cal_p2[dit*6:(dit+1)*6] = mc_p2

        for fdx, file in enumerate(sg_files):
            sg_visphi_p1[fdx] = np.angle((np.exp(1j*sg_visphi_p1[fdx]/180*np.pi)
                                          / cal_p1), deg=True)
            sg_visphi_p2[fdx] = np.angle((np.exp(1j*sg_visphi_p2[fdx]/180*np.pi)
                                          / cal_p2), deg=True)

        for fdx, file in enumerate(s2_files):
            s2_visphi_p1[fdx] = np.angle((np.exp(1j*s2_visphi_p1[fdx]/180*np.pi)
                                          / cal_p1), deg=True)
            s2_visphi_p2[fdx] = np.angle((np.exp(1j*s2_visphi_p2[fdx]/180*np.pi)
                                          / cal_p2), deg=True)

        ##########################
        # Poscor
        ##########################
        if poscor:
            s2_B = np.zeros((s2_u.shape[0], 6, nchannel, 2))
            SpFreq = np.zeros((s2_u.shape[0], 6, nchannel))
            for bl in range(6):
                if ndit == 1:
                    s2_B[:, bl, :, 0] = s2_u[:, bl, :]
                    s2_B[:, bl, :, 1] = s2_v[:, bl, :]
                    SpFreq[:, bl, :] = np.sqrt(s2_u[:, bl, :]**2
                                               + s2_v[:, bl, :]**2)
                else:
                    s2_u[s2_u == 0] = np.nan
                    s2_v[s2_v == 0] = np.nan
                    s2_B[:, bl, :, 0] = np.nanmean(s2_u[:, bl::6, :], 1)
                    s2_B[:, bl, :, 1] = np.nanmean(s2_v[:, bl::6, :], 1)
                    SpFreq[:, bl, :] = np.sqrt(np.nanmean(s2_u[:, bl::6, :], 1)**2
                                               + np.nanmean(s2_v[:, bl::6, :], 1)**2)

            s2_dB = np.copy(s2_B)
            s2_dB = s2_dB - s2_B[calib_file]

            dB1 = np.transpose([s2_dB[:, :, :, 0].flatten(),
                                s2_dB[:, :, :, 1].flatten()])

            nfiles = len(s2_visphi_p1)
            s2_visphi_fit = np.zeros((nfiles, 6, nchannel))
            s2_visphi_err_fit = np.zeros((nfiles, 6, nchannel))
            if ndit == 1:
                for bl in range(6):
                    s2_visphi_fit[:, bl, :] = s2_visphi_p1[:, bl, :]
                    s2_visphi_fit[:, bl, :] += s2_visphi_p2[:, bl, :]
                    s2_visphi_err_fit[:, bl, :] = (np.sqrt(s2_visphi_err_p1[:, bl, :]**2
                                                           + s2_visphi_err_p2[:, bl, :]**2)
                                                   / np.sqrt(2))
            else:
                for bl in range(6):
                    s2_visphi_fit[:, bl, :] = np.nanmean(s2_visphi_p1[:, bl::6, :],1)
                    s2_visphi_fit[:, bl, :] += np.nanmean(s2_visphi_p2[:, bl::6, :],1)
                    s2_visphi_err_fit[:, bl, :] = (np.sqrt(np.nanmean(s2_visphi_err_p1[:, bl::6, :], 1)**2
                                                           + np.nanmean(s2_visphi_err_p2[:, bl::6, :], 1)**2)
                                                   / np.sqrt(2))

            s2_visphi_fit /= 2
            s2_visphi_fit[:, :, :2] = np.nan
            s2_visphi_fit[:, :, -2:] = np.nan

            Vphi_err = s2_visphi_err_fit.flatten()
            Vphi = s2_visphi_fit.flatten()

            Vphi2 = Vphi[~np.isnan(Vphi)]/360
            Vphi_err2 = Vphi_err[~np.isnan(Vphi)]/360
            dB2 = np.zeros((len(Vphi2),2))
            dB2[:, 0] = dB1[~np.isnan(Vphi), 0]
            dB2[:, 1] = dB1[~np.isnan(Vphi), 1]

            def f(dS):
                Chi = (Vphi2-np.dot(dB2, dS)) / (Vphi_err2)
                return Chi

            try:
                dS, pcov, infodict, errmsg, success = optimize.leastsq(f, x0=[0, 0], full_output=1)
            except TypeError:
                print('PosCor failed')
                dS = [0, 0]

            if self.verbose:
                print('Applied poscor: (%.3f,%.3f) mas ' % (dS[0], dS[1]))
                print('Chi2 of poscor: %.2f \n' % np.sum(f(dS)**2))
            self.dS = dS

            if plot:
                n = nfiles
                par = np.linspace(0, np.max(s2_t)+10, 100)
                norm = matplotlib.colors.Normalize(vmin=np.min(par),
                                                   vmax=np.max(par))
                c_m = plt.cm.inferno
                s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
                s_m.set_array([])

                fitres = np.dot(dB1, dS)*360
                fitres_r = np.reshape(fitres, (n, 6, len(SpFreq[0, 0])))
                for idx in range(n):
                    for bl in range(6):
                        if ndit == 1:
                            _plott = s2_t[idx]
                        else:
                            _plott = np.mean(s2_t[idx])
                        plt.errorbar(SpFreq[idx, bl, 2:-2] * 1000,
                                     s2_visphi_fit[idx, bl, 2:-2],
                                     s2_visphi_err_fit[idx, bl, 2:-2],
                                     ls='', marker='o',
                                     color=s_m.to_rgba(_plott), alpha=0.5)
                        plt.plot(SpFreq[idx, bl]*1000, fitres_r[idx, bl],
                                 color=s_m.to_rgba(_plott))
                plt.ylim(-74, 74)
                plt.colorbar(s_m, label='Time [min]')
                plt.xlabel('Spatial frequency [1/as]')
                plt.ylabel('Visibility phase [deg]')
                plt.title('Poscor:  (%.3f,%.3f) mas ' % (dS[0], dS[1]))
                plt.show()

            s2_B = np.zeros((s2_u.shape[0], ndit*6, nchannel, 2))
            s2_B[:, :, :, 0] = s2_u
            s2_B[:, :, :, 1] = s2_v
            s2_dB = np.copy(s2_B)

            B_calib = np.zeros((6, nchannel, 2))
            for bl in range(6):
                B_calib[bl] = np.nanmean(s2_B[calib_file][bl::6], 0)
                s2_dB[:, bl::6, :, :] = s2_dB[:, bl::6, :, :] - B_calib[bl]

            sg_B = np.zeros((sg_u.shape[0], ndit*6, nchannel, 2))
            sg_B[:, :, :, 0] = sg_u
            sg_B[:, :, :, 1] = sg_v
            sg_dB = np.copy(sg_B)

            B_calib = np.zeros((6, nchannel, 2))
            for bl in range(6):
                B_calib[bl] = np.nanmean(s2_B[calib_file][bl::6], 0)
                sg_dB[:, bl::6, :, :] = sg_dB[:, bl::6, :, :] - B_calib[bl]

            sg_visphi_p1 -= np.dot(sg_dB, dS)*360
            sg_visphi_p2 -= np.dot(sg_dB, dS)*360

            s2_visphi_p1 -= np.dot(s2_dB, dS)*360
            s2_visphi_p2 -= np.dot(s2_dB, dS)*360

        ##########################
        # linear bl fit
        ##########################
        if linear_cor:
            def linreg(x, a, b):
                return a*x + b

            for bl in range(6):
                x = sg_t[fitstart:fitstop].flatten()
                y = np.nanmean(sg_visphi_p1[fitstart:fitstop, bl::6, 2:-2],2).flatten()
                yerr = np.mean(sg_visphi_err_p1[fitstart:fitstop, bl::6, 2:-2],2).flatten()
                valid = ~(np.isnan(x) | np.isnan(y))
                popt, pcov = optimize.curve_fit(linreg, x[valid], y[valid],
                                                sigma=yerr[valid], p0=[0, 0])
                for wdx in range(len(wave)):
                    sg_visphi_p1[:, bl::6, wdx] -= linreg(sg_t, *popt)

                y = np.nanmean(sg_visphi_p2[fitstart:fitstop, bl::6, 2:-2], 2).flatten()
                yerr = np.mean(sg_visphi_err_p2[fitstart:fitstop, bl::6, 2:-2], 2).flatten()
                valid = ~(np.isnan(x) | np.isnan(y))
                popt, pcov = optimize.curve_fit(linreg, x[valid], y[valid],
                                                sigma=yerr[valid], p0=[0, 0])
                for wdx in range(len(wave)):
                    sg_visphi_p2[:, bl::6, wdx] -= linreg(sg_t, *popt)

        ##########################
        # unwrap phases
        ##########################
        sg_visphi_p1 = ((sg_visphi_p1+180) % 360) - 180
        sg_visphi_p2 = ((sg_visphi_p2+180) % 360) - 180
        s2_visphi_p1 = ((s2_visphi_p1+180) % 360) - 180
        s2_visphi_p2 = ((s2_visphi_p2+180) % 360) - 180

        result = [[sg_t, sg_lst, sg_ang, sg_visphi_p1, sg_visphi_err_p1,
                   sg_visphi_p2, sg_visphi_err_p2],
                  [s2_t, s2_lst, s2_ang, s2_visphi_p1, s2_visphi_err_p1,
                   s2_visphi_p2, s2_visphi_err_p2]]
        self.sg_visphi_p1 = sg_visphi_p1
        self.sg_visphi_p2 = sg_visphi_p2

        if plot and mode is not None:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()
            for idx in range(6):
                all_t = np.concatenate((sg_lst.flatten(), s2_lst.flatten()))
                all_t_s = all_t.argsort()
                all_cor = np.concatenate((np.nanmedian(sg_correction[:,idx::6,2:-2],2).flatten(), np.nanmedian(s2_correction[:,idx::6,2:-2],2).flatten()))
                all_cor_wo = np.concatenate((np.nanmedian(sg_correction_wo[:,idx::6,2:-2],2).flatten(), 
                                             np.nanmedian(s2_correction_wo[:,idx::6,2:-2],2).flatten()))
                all_cor = -all_cor[all_t_s]
                all_cor_wo = -all_cor_wo[all_t_s]
                all_t = all_t[all_t_s]

                if idx == 3:
                    ax1.plot(s2_lst.flatten(), (np.nanmedian(s2_visphi_p1[:,idx::6,2:-2],2).flatten()+
                                              np.nanmedian(s2_visphi_p1[:,idx::6,2:-2],2).flatten())/2+idx*50, 
                             ls='', lw=0.5, marker='D', zorder=10, color=colors_baseline[idx],
                             markersize=2, alpha=0.5, markeredgecolor='k', label='S2 files')
                    ax1.plot(sg_lst.flatten(), (np.nanmedian(sg_visphi_p1[:,idx::6,2:-2],2).flatten()+
                                              np.nanmedian(sg_visphi_p2[:,idx::6,2:-2],2).flatten())/2+idx*50, 
                             ls='', lw=0.5, marker='o', zorder=10, color=colors_baseline[idx],
                             markersize=2, alpha=0.5, label='Sgr A* files')
                    ax1.plot(all_t, all_cor+idx*50, color='r', zorder=11, alpha=0.8, label='Correction')
                    if cor_cor:
                        ax1.plot(all_t, all_cor_wo+idx*50, color='k', ls='--', zorder=11, alpha=0.8, 
                                label='Correction w/o phasecor')
                else:
                    ax1.plot(s2_lst.flatten(), (np.nanmedian(s2_visphi_p1[:,idx::6,2:-2],2).flatten()+
                                              np.nanmedian(s2_visphi_p1[:,idx::6,2:-2],2).flatten())/2+idx*50, 
                             ls='', lw=0.5, marker='D', zorder=10, color=colors_baseline[idx],
                             markersize=2, alpha=0.5, markeredgecolor='k')
                    ax1.plot(sg_lst.flatten(), (np.nanmedian(sg_visphi_p1[:,idx::6,2:-2],2).flatten()+
                                              np.nanmedian(sg_visphi_p2[:,idx::6,2:-2],2).flatten())/2+idx*50, 
                             ls='', lw=0.5, marker='o', zorder=10, color=colors_baseline[idx],
                             markersize=2, alpha=0.5)
                    ax1.plot(all_t, all_cor+idx*50, color='r', zorder=11, alpha=0.8)
                    if cor_cor:
                        ax1.plot(all_t, all_cor_wo+idx*50, color='k', ls='--', zorder=11, alpha=0.8)
                ax1.axhline(idx*50, lw=0.5, color=colors_baseline[idx])
            if linear_cor:
                ax1.axvline(np.nanmean(sg_t, 1)[fitstart], ls='--', color='grey', lw=0.5)
                ax1.axvline(np.nanmean(sg_t, 1)[fitstop-1], ls='--', color='grey', lw=0.5)
            ax1.set_xlim(15,21)
            locs = ax1.get_xticks()
            labe = ax1.get_xticklabels()
            all_lst = np.concatenate((sg_lst.flatten(), s2_lst.flatten()))
            all_ang = np.concatenate((sg_ang.flatten(), s2_ang.flatten()))
            ang_loc = []
            for loc in locs:
                ang_loc.append("%i" % all_ang[find_nearest(loc, all_lst)])

            ax_top_Ticks = ang_loc
            ax2.set_xticks(locs)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticklabels(ang_loc)
            ax2.set_xbound(ax1.get_xbound())
            ax2.set_xlabel('Ref. Angle [deg]')

            ax1.legend()
            if ndit == 1:
                plt.ylabel('Phase [deg]')
            else:
                plt.ylabel('Phase [deg]\n(Median per exposure)')
            ax1.set_xlabel('LST [h]')
            plt.ylim(-79,299)
            # plt.xlim(-5, xmax)
            plt.show()

        ##########################
        # save data
        ##########################
        if save:
            if correction:
                if type(save) == str:
                    folder = file[:file.find('GRAVI')] + save
                    if self.verbose:
                        print("saving folder: %s" % folder)
                else:
                    if self.verbose:
                        print("saving to default location")
                    if linear_cor:
                        folder = file[:file.find('GRAVI')] + 'poscor_met_' + mode +'_lincor/'
                    else:
                        folder = file[:file.find('GRAVI')] + 'poscor_met_' + mode +'/'
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                if self.verbose:
                    print('Save files in: %s' % folder)
                for fdx, file in enumerate(sg_files):
                    fname = file[file.find('GRAVI'):]
                    visphi_p1 = result[0][3][fdx]
                    visphi_p2 = result[0][5][fdx]

                    if np.isnan(visphi_p1).all():
                        print('%s is all nan' % fname)
                    else:
                        d = fits.open(file)
                        d['OI_VIS', 11].data['VISPHI'] = visphi_p1
                        d['OI_VIS', 12].data['VISPHI'] = visphi_p2
                        d.writeto(folder+fname, overwrite=True)

                for fdx, file in enumerate(s2_files):
                    fname = file[file.find('GRAVI'):]
                    visphi_p1 = result[1][3][fdx]
                    visphi_p2 = result[1][5][fdx]
                    if np.isnan(visphi_p1).all():
                        print('%s is all nan' % fname)
                    else:
                        d = fits.open(file)
                        d['OI_VIS', 11].data['VISPHI'] = visphi_p1
                        d['OI_VIS', 12].data['VISPHI'] = visphi_p2
                        d.writeto(folder+fname, overwrite=True)

            else:
                if type(save) == str:
                    folder = file[:file.find('GRAVI')] + save
                    if self.verbose:
                        print("saving folder: %s" % folder)
                else:
                    if self.verbose:
                        print("saving to default location")
                    if poscor:
                        if linear_cor:
                            folder = file[:file.find('GRAVI')] + 'poscor_nometcor_new/'
                        else:
                            folder = file[:file.find('GRAVI')] + 'poscor_nometcor/'
                    else:
                        if linear_cor:
                            folder = file[:file.find('GRAVI')] + 'nometcor_new/'
                        else:
                            folder = file[:file.find('GRAVI')] + 'nometcor/'
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                for fdx, file in enumerate(sg_files):
                    fname = file[file.find('GRAVI'):]
                    visphi_p1 = result[0][3][fdx]
                    visphi_p2 = result[0][5][fdx]

                    if np.isnan(visphi_p1).all():
                        print('%s is all nan' % fname)
                    else:
                        d = fits.open(file)
                        d['OI_VIS', 11].data['VISPHI'] = visphi_p1
                        d['OI_VIS', 12].data['VISPHI'] = visphi_p2
                        d.writeto(folder+fname, overwrite=True)

                for fdx, file in enumerate(s2_files):
                    fname = file[file.find('GRAVI'):]
                    visphi_p1 = result[1][3][fdx]
                    visphi_p2 = result[1][5][fdx]
                    if np.isnan(visphi_p1).all():
                        print('%s is all nan' % fname)
                    else:
                        d = fits.open(file)
                        d['OI_VIS', 11].data['VISPHI'] = visphi_p1
                        d['OI_VIS', 12].data['VISPHI'] = visphi_p2
                        d.writeto(folder+fname, overwrite=True)

            self.savefolder = folder
        self.alldata = result
        if ret:
            return self.alldata

    def calibrate_all(self, mode, allS2=False, *args, **kwargs):
        self.process_night(mode, *args, **kwargs)
        sf = self.savefolder
        cf = self.savefolder + 'calibrated_oneS2'
        if allS2:
            os.system('run_gravi_recalibrate.py %s %s -c=S2 -s=S2 -s=SGRA'
                      % (sf, cf))
        else:
            os.system('run_gravi_recalibrate.py %s %s -c=%s -s=S2 -s=SGRA'
                      % (sf, cf, self.calibrator))