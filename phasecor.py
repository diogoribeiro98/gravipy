import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pkg_resources import resource_filename
import pandas as pd
from astropy.io import fits
from scipy import interpolate, optimize
from astropy.time import Time
from datetime import timedelta, datetime
from joblib import Parallel, delayed
import multiprocessing
import os
import sys
import emcee
import corner
import glob
import gc

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




########################
# Auxilary functions

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def averaging(x, N):
    if x.ndim == 2:
        res = np.zeros((x.shape[0], x.shape[1]//N))
        for idx in range(x.shape[0]):
            res[idx] = np.nanmean(x[idx].reshape(-1, N), axis=1)
        return res
    elif x.ndim == 1:
        # pad with nans
        l = x.shape[0]
        xx = np.pad(x, (0,N-l%N), constant_values=np.nan)
        res = np.nanmean(xx.reshape(-1, N), axis=1)
        return res
    
def averaging_std(x, N):
    if x.ndim == 2:
        res = np.zeros((x.shape[0], x.shape[1]//N))
        for idx in range(x.shape[0]):
            res[idx] = np.nanstd(x[idx].reshape(-1, N), axis=1)
        return res
    elif x.ndim == 1:
        # pad with nans
        l = x.shape[0]
        xx = np.pad(x, (0,N-l%N), constant_values=np.nan)
        res = np.nanstd(xx.reshape(-1, N), axis=1)
        return res    
    
def get_angle_header_all(header, tel, length):
    pa1 = header["ESO ISS PARANG START"]
    pa2 = header["ESO ISS PARANG END"]
    parang = np.linspace(pa1, pa2, length+1)
    parang = parang[:-1]
    drottoff = header["ESO INS DROTOFF" +str(4-tel)]
    dx = header["ESO INS SOBJ X"] - header["ESO INS SOBJ OFFX"]
    dy = header["ESO INS SOBJ Y"] - header["ESO INS SOBJ OFFY"]
    posangle = np.arctan2(dx, dy) * 180 / np.pi;
    fangle = - posangle - drottoff + 270
    angle = fangle + parang + 45.
    return (angle-180)%360

def get_angle_header_start(header, tel):
    pa1 = header["ESO ISS PARANG START"]
    parang = pa1
    drottoff = header["ESO INS DROTOFF" +str(4-tel)]
    dx = header["ESO INS SOBJ X"] - header["ESO INS SOBJ OFFX"]
    dy = header["ESO INS SOBJ Y"] - header["ESO INS SOBJ OFFY"]
    posangle = np.arctan2(dx, dy) * 180 / np.pi;
    fangle = - posangle - drottoff + 270
    angle = fangle + parang + 45.
    return (angle-180)%360

def get_angle_header_mean(header, tel):
    pa1 = header["ESO ISS PARANG START"]
    pa2 = header["ESO ISS PARANG END"]
    parang = (pa1+pa2)/2
    drottoff = header["ESO INS DROTOFF" +str(4-tel)]
    dx = header["ESO INS SOBJ X"] - header["ESO INS SOBJ OFFX"]
    dy = header["ESO INS SOBJ Y"] - header["ESO INS SOBJ OFFY"]
    posangle = np.arctan2(dx, dy) * 180 / np.pi;
    fangle = - posangle - drottoff + 270
    angle = fangle + parang + 45.
    return (angle-180)%360

def rotation(ang):
    return np.array([[np.cos(ang), np.sin(ang)],
                     [-np.sin(ang), np.cos(ang)]])
    
def convert_date(date):
    t = Time(date)
    t2 = Time('2000-01-01T12:00:00')
    date_decimal = (t.mjd - t2.mjd)/365.25+2000
    
    date = date.replace('T', ' ')
    date = date.split('.')[0]
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return date_decimal, date



#########################
# Read in metrology correction

def read_correction(mcor_files, xscale, list_dim=1, fancy=True, wrap=False, lst=False,
                    textpos=15, av=20, std_cut=0.03, met_cut=0.2, bequiet=False):
    """
    Reads in the correction created by 
    TODO add function to create correction
    
    Input:
    mcor_files:     List of all metrology measurements
    xscale:         X axis of tje mcor_files
    list_dim:       If mcor_files is a list of lists this has to be length of that lists [1]
    fancy:          Use the fancy median correction [True]
    wrap:           Wraps the metrology at 180 degree [False]
    lst:            Has to be true if input is a function of lst [False]
    textpos:        Postition to plot text in figures [15]
    av:             Average for the median of the output figure [20]
    std_cut:        Metrology datapoints are ignored if the std is higher than this value [0.03]
    met_cut:        Metrology datapoints are ignored if the value is higher than this [0.2]
    bequiet:        If true all outputs are supressed [False]
    
    Output:
    x values and array with all corrected metrology measurements
    """
    
    if list_dim == 1:
        if len(mcor_files) < 10 and not not_fancy:
            print('Very short list of input.')
            print('If you want to use several lists, use list_dim != 1.')
            print('Else not_fancy=True shoule be better...')
        if wrap:
            if lst:
                raise ValueError('Wrap and lst do not work together!')
            def wrap_data(a):
                a1 = a[:a.shape[0]//2]
                a2 = a[a.shape[0]//2+1:]
                b = np.nanmean((a1,a2),0)
                return b
            TELFC_MCORR_S2 = np.array([wrap_data(np.load(fi)*1e6) for fi in mcor_files])
            x = np.load(xscale)[:-1]
            x = x[:len(x)//2]
        else:
            TELFC_MCORR_S2 = np.array([np.load(fi)*1e6 for fi in mcor_files])
            x = np.load(xscale)[:-1]

        ndata = len(mcor_files)
        colors = plt.cm.jet(np.linspace(0,1,ndata))
        len_data = []
        ndata = len(TELFC_MCORR_S2)
        for idx in range(ndata):
            len_data.append(len(TELFC_MCORR_S2[idx,0,~np.isnan(TELFC_MCORR_S2[idx,0])]))
        sort = np.array(len_data).argsort()[::-1]
        TELFC_MCORR_S2 = TELFC_MCORR_S2[sort]
        
    else:
        if len(mcor_files) != list_dim:
            raise ValueError('list_dim is different then length of input list, input is wrong')
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
                    len_data.append(len(in_mcorr[idx,0,~np.isnan(in_mcorr[idx,0])]))
                sort = np.array(len_data).argsort()[::-1]
                if ldx == 0:
                    TELFC_MCORR_S2 = in_mcorr[sort]
                else:
                    TELFC_MCORR_S2 = np.concatenate((TELFC_MCORR_S2, in_mcorr[sort]), 0)
            x = np.load(xscale)[:-1]
            ndata = len(TELFC_MCORR_S2)
            colors = plt.cm.jet(np.linspace(0,1,ndata))
            colors_lists = ['k', color2, color3]

    if not bequiet:
        gs = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        plt.figure(figsize=(8,6))
        for tel in range(4):
            ax = plt.subplot(gs[tel%2,tel//2])
            for idx in range(ndata):
                data = TELFC_MCORR_S2[idx,:,tel] - np.nanmean(TELFC_MCORR_S2[idx,:,tel])
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
            if tel%2 != 1:
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
                TELFC_MCORR_S2_corr[idx,:,tel] -= np.nanmean(TELFC_MCORR_S2_corr[idx,:,tel])

        for tel in range(4):
            TELFC_MCORR_S2_corr[:,:,tel] -= np.nanmean(TELFC_MCORR_S2_corr[:,:,tel])
                
        gs = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        plt.figure(figsize=(8,6))
        for tel in range(4):
            ax = plt.subplot(gs[tel%2,tel//2])
            for idx in range(ndata):
                data = TELFC_MCORR_S2_corr[idx,:,tel]
                if list_dim == 1:
                    plt.plot(x, data, marker='.', ls='', alpha=0.5, color='k')
                else:
                    ndx = np.where(np.array(len_lists) <= idx)[0][-1]
                    plt.plot(angl, data, marker='.', ls='', 
                             alpha=0.1, color=colors_lists[ndx])
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3,0.3)
            plt.plot(averaging(x,av), averaging(np.nanmedian(TELFC_MCORR_S2_corr[:,:,tel], 0), av), color=color1)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel%2 != 1:
                ax.set_xticklabels([])
            else:
                if lst:
                    plt.xlabel('LST [h]')
                else:
                    plt.xlabel('Ref. angle [deg]')
        plt.show()

        return x, TELFC_MCORR_S2_corr

    TELFC_MCORR_S2_corr = np.copy(TELFC_MCORR_S2)
    TELFC_MCORR_S2_av = np.zeros((ndata,TELFC_MCORR_S2.shape[1]//av,4))
    TELFC_MCORR_S2_std = np.zeros((ndata,TELFC_MCORR_S2.shape[1]//av,4))
    for idx in range(ndata):
        for tel in range(4):
            TELFC_MCORR_S2_av[idx,:,tel] = averaging(TELFC_MCORR_S2[idx,:,tel],av)[:-1]
            TELFC_MCORR_S2_std[idx,:,tel] = averaging_std(TELFC_MCORR_S2[idx,:,tel],av)[:-1]
    for tel in range(4):
        TELFC_MCORR_S2_corr[0,:,tel] -= np.nanmean(TELFC_MCORR_S2_corr[0,:,tel])
        TELFC_MCORR_S2_av[0,:,tel] -= np.nanmean(TELFC_MCORR_S2_av[0,:,tel])
        TELFC_MCORR_S2_corr[0,np.where(np.abs(TELFC_MCORR_S2_corr[0,:,tel]) > met_cut),tel] *= np.nan
        TELFC_MCORR_S2_av[0,np.where(np.abs(TELFC_MCORR_S2_av[0,:,tel]) > met_cut),tel] *= np.nan

    for idx in range(1,ndata):
        if idx == 1:
            corr = TELFC_MCORR_S2_av[0]
        else:
            corr = np.nanmedian(TELFC_MCORR_S2_av[:(idx-1)],0)

        for tel in range(4):
            TELFC_MCORR_S2_av[idx,[np.where(TELFC_MCORR_S2_std[idx,:,tel] > std_cut)],tel] = np.nan
            mask = ~np.isnan(corr[:,tel])*~np.isnan(TELFC_MCORR_S2_av[idx,:,tel])
            if len(mask[mask == True]) == 0:
                TELFC_MCORR_S2_av[idx,:,tel] -= np.nanmedian(TELFC_MCORR_S2_av[idx,:,tel])
                TELFC_MCORR_S2_corr[idx,:,tel] -= np.nanmedian(TELFC_MCORR_S2_corr[idx,:,tel])
                if not bequiet:
                    if tel == 0:
                        print('%i no overlap' % idx)
            else:
                mdiff = np.mean(TELFC_MCORR_S2_av[idx,mask,tel])-np.mean(corr[mask,tel])
                TELFC_MCORR_S2_corr[idx,:,tel] -= mdiff
                TELFC_MCORR_S2_av[idx,:,tel] -= mdiff
                if not bequiet:
                    if tel == 0:
                        print(idx, mdiff)
            if np.nanstd(TELFC_MCORR_S2_corr[idx,:,tel]) > 0.1:
                if not bequiet:
                    print('%i, %i: std of data too big' % (idx, tel))
                TELFC_MCORR_S2_corr[idx,:,tel] *= np.nan
                TELFC_MCORR_S2_av[idx,:,tel] *= np.nan

            TELFC_MCORR_S2_corr[idx,np.where(np.abs(TELFC_MCORR_S2_corr[idx,:,tel]) > met_cut),tel] *= np.nan
            TELFC_MCORR_S2_av[idx,np.where(np.abs(TELFC_MCORR_S2_av[idx,:,tel]) > met_cut),tel] *= np.nan
            
    if not bequiet:
        gs = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        plt.figure(figsize=(8,6))
        for tel in range(4):
            ax = plt.subplot(gs[tel%2,tel//2])
            for idx in range(ndata):
                data = TELFC_MCORR_S2_corr[idx,:,tel]
                plt.plot(x, data,
                         marker='.', ls='', alpha=0.5, color=colors[idx])
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3,0.3)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel%2 != 1:
                ax.set_xticklabels([])
            else:
                if lst:
                    plt.xlabel('LST [h]')
                else:
                    plt.xlabel('Ref. angle [deg]')
        plt.show()
    
    for tel in range(4):
        TELFC_MCORR_S2_corr[:,:,tel] -= np.nanmean(TELFC_MCORR_S2_corr[:,:,tel])
    
    if not bequiet:
        gs = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        plt.figure(figsize=(8,6))
        for tel in range(4):
            ax = plt.subplot(gs[tel%2,tel//2])
            for idx in range(ndata):
                data = TELFC_MCORR_S2_corr[idx,:,tel]
                if list_dim == 1:
                    plt.plot(x, data, marker='.', ls='', alpha=0.5, color='k')
                else:
                    ndx = np.where(np.array(len_lists) <= idx)[0][-1]
                    plt.plot(x, data, marker='.', ls='', 
                            alpha=0.1, color=colors_lists[ndx])
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3,0.3)
            plt.plot(averaging(x,av), averaging(np.nanmedian(TELFC_MCORR_S2_corr[:,:,tel], 0), av), color=color1)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel%2 != 1:
                ax.set_xticklabels([])
            else:
                if lst:
                    plt.xlabel('LST [h]')
                else:
                    plt.xlabel('Ref. angle [deg]')
        plt.show()

    return x, TELFC_MCORR_S2_corr


def read_correction_sebo(sebo_mcor_files, sebo_angl_files, fancy=True, bequiet=False,
                        wrap=False, textpos=15, av=20, std_cut=0.03, met_cut=0.2):
    """
    Same as read_correction, but for files created by Sebastiano Daniel Maximilian von Fellenberg
    """
    if wrap:
        def wrap_data(a):
            a1 = a[:,:a.shape[1]//2]
            a2 = a[:,a.shape[1]//2+1:]
            b = np.nanmean((a1,a2),0)
            return b
        TELFC_MCORR_S2 = np.array([wrap_data(np.load(fi)*1e6) for fi in sebo_mcor_files])
        ANGLE_S2 = np.load(sebo_angl_files[0])[:-1]
        ANGLE_S2 = ANGLE_S2[:len(ANGLE_S2)//2]
    else:
        TELFC_MCORR_S2 = np.array([np.load(fi)*1e6 for fi in sebo_mcor_files])
        ANGLE_S2 = np.load(sebo_angl_files[0])[:-1]
        
    ndata = len(sebo_mcor_files)
    colors = plt.cm.jet(np.linspace(0,1,ndata))
    len_data = []
    ndata = len(TELFC_MCORR_S2)
    for idx in range(ndata):
        len_data.append(len(TELFC_MCORR_S2[idx,0,~np.isnan(TELFC_MCORR_S2[idx,0])]))
    sort = np.array(len_data).argsort()[::-1]
    TELFC_MCORR_S2 = TELFC_MCORR_S2[sort]
    if not bequiet:
        gs = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        plt.figure(figsize=(8,6))
        for tel in range(4):
            ax = plt.subplot(gs[tel%2,tel//2])
            for idx in range(ndata):
                angl = ANGLE_S2
                data = TELFC_MCORR_S2[idx,tel] - np.nanmean(TELFC_MCORR_S2[idx,tel])
                plt.plot(angl, data,
                         marker='.', ls='', alpha=0.5, color=colors[idx])
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3,0.3)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel%2 != 1:
                ax.set_xticklabels([])
            else:
                plt.xlabel('Ref. angle [deg]')
        plt.show()

    if not fancy:
        TELFC_MCORR_S2_corr = np.copy(TELFC_MCORR_S2)
        for idx in range(ndata):
            for tel in range(4):
                TELFC_MCORR_S2_corr[idx,tel] -= np.nanmean(TELFC_MCORR_S2_corr[idx,tel])
        
        for tel in range(4):
            TELFC_MCORR_S2_corr[:,tel] -= np.nanmean(TELFC_MCORR_S2_corr[:,tel])
                
        gs = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        plt.figure(figsize=(8,6))
        for tel in range(4):
            ax = plt.subplot(gs[tel%2,tel//2])
            for idx in range(ndata):
                angl = ANGLE_S2
                data = TELFC_MCORR_S2_corr[idx,tel]
                plt.plot(angl, data, marker='.', ls='', alpha=0.5, color='k')
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3,0.3)

            plt.plot(averaging(ANGLE_S2,av), averaging(np.nanmedian(TELFC_MCORR_S2_corr[:,tel], 0), av), color=color1)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel%2 != 1:
                ax.set_xticklabels([])
            else:
                plt.xlabel('Ref. angle [deg]')
        plt.show()

        return ANGLE_S2, TELFC_MCORR_S2_corr
    
    TELFC_MCORR_S2_corr = np.copy(TELFC_MCORR_S2)
    TELFC_MCORR_S2_av = np.zeros((ndata,4,TELFC_MCORR_S2.shape[2]//av))
    TELFC_MCORR_S2_std = np.zeros((ndata,4,TELFC_MCORR_S2.shape[2]//av))
    for idx in range(ndata):
        for tel in range(4):
            TELFC_MCORR_S2_av[idx,tel] = averaging(TELFC_MCORR_S2[idx,tel],av)[:-1]
            TELFC_MCORR_S2_std[idx,tel] = averaging_std(TELFC_MCORR_S2[idx,tel],av)[:-1]
    for tel in range(4):
        TELFC_MCORR_S2_corr[0,tel] -= np.nanmean(TELFC_MCORR_S2_corr[0,tel])
        TELFC_MCORR_S2_av[0,tel] -= np.nanmean(TELFC_MCORR_S2_av[0,tel])
        TELFC_MCORR_S2_corr[0,tel,np.where(np.abs(TELFC_MCORR_S2_corr[0,tel]) > met_cut)] *= np.nan
        TELFC_MCORR_S2_av[0,tel,np.where(np.abs(TELFC_MCORR_S2_av[0,tel]) > met_cut)] *= np.nan

    for idx in range(1,ndata):
        if idx == 1:
            corr = TELFC_MCORR_S2_av[0]
        else:
            corr = np.nanmedian(TELFC_MCORR_S2_av[:(idx-1)],0)

        for tel in range(4):
            TELFC_MCORR_S2_av[idx,tel,[np.where(TELFC_MCORR_S2_std[idx,tel] > std_cut)]] = np.nan
            mask = ~np.isnan(corr[tel])*~np.isnan(TELFC_MCORR_S2_av[idx,tel])
            if len(mask[mask == True]) == 0:
                TELFC_MCORR_S2_av[idx,tel] -= np.nanmedian(TELFC_MCORR_S2_av[idx,tel])
                TELFC_MCORR_S2_corr[idx,tel] -= np.nanmedian(TELFC_MCORR_S2_corr[idx,tel])
                if not bequiet:
                    if tel == 0:
                        print('%i no overlap' % idx)
            else:
                mdiff = np.mean(TELFC_MCORR_S2_av[idx,tel,mask])-np.mean(corr[tel,mask])
                TELFC_MCORR_S2_corr[idx,tel] -= mdiff
                TELFC_MCORR_S2_av[idx,tel] -= mdiff
                if not bequiet:
                    if tel == 0:
                        print(idx, mdiff)
            if np.nanstd(TELFC_MCORR_S2_corr[idx,tel]) > 0.1:
                if not bequiet:
                    print('%i, %i: std of data too big' % (idx, tel))
                TELFC_MCORR_S2_corr[idx,tel] *= np.nan
                TELFC_MCORR_S2_av[idx,tel] *= np.nan

            TELFC_MCORR_S2_corr[idx,tel,np.where(np.abs(TELFC_MCORR_S2_corr[idx,tel]) > met_cut)] *= np.nan
            TELFC_MCORR_S2_av[idx,tel,np.where(np.abs(TELFC_MCORR_S2_av[idx,tel]) > met_cut)] *= np.nan
            
    if not bequiet:
        gs = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
        plt.figure(figsize=(8,6))

        for tel in range(4):
            ax = plt.subplot(gs[tel%2,tel//2])
            for idx in range(ndata):
                angl = ANGLE_S2
                data = TELFC_MCORR_S2_corr[idx,tel]
                plt.plot(angl, data,
                         marker='.', ls='', alpha=0.5, color=colors[idx])
            plt.text(textpos, -0.25, 'UT%i' % (4-tel))
            plt.ylim(-0.3,0.3)
            if tel//2 != 0:
                ax.set_yticklabels([])
            else:
                plt.ylabel('TELFC_MCORR [$\mu$m]')
            if tel%2 != 1:
                ax.set_xticklabels([])
            else:
                plt.xlabel('Ref. angle [deg]')
        plt.show()

    for tel in range(4):
        TELFC_MCORR_S2_corr[:,tel] -= np.nanmean(TELFC_MCORR_S2_corr[:,tel])

    gs = gridspec.GridSpec(2,2,wspace=0.1,hspace=0.1)
    plt.figure(figsize=(8,6))
    for tel in range(4):
        ax = plt.subplot(gs[tel%2,tel//2])
        for idx in range(ndata):
            angl = ANGLE_S2
            data = TELFC_MCORR_S2_corr[idx,tel]
            plt.plot(angl, data, marker='.', ls='', alpha=0.5, color='k')
        plt.text(textpos, -0.25, 'UT%i' % (4-tel))
        plt.ylim(-0.3,0.3)

        plt.plot(averaging(ANGLE_S2,av), averaging(np.nanmedian(TELFC_MCORR_S2_corr[:,tel], 0), av), color=color1)
        if tel//2 != 0:
            ax.set_yticklabels([])
        else:
            plt.ylabel('TELFC_MCORR [$\mu$m]')
        if tel%2 != 1:
            ax.set_xticklabels([])
        else:
            plt.xlabel('Ref. angle [deg]')
    plt.show()

    return ANGLE_S2, TELFC_MCORR_S2_corr


######################
# Correct a set of data


def correct_data(files, mode, subspacing=1, plotav=8, plot=False, lstplot=False):
    corrections_dict = get_corrections(bequiet=True)
    try:
        interp_list = corrections_dict[mode]
    except KeyError:
        print('Given mode not avialable, use one of those:')
        print(corrections_dict.keys())
        raise KeyError('Given mode not avialable')
        
    if 'lst' in mode:
        lst_corr = True
        print('Apply LST correction')
    else:
        lst_corr = False
        
    if 'bl' in mode:
        bl_corr = True
        print('Apply BL correction')
    else:
        bl_corr = False
        
    bl_array = np.array([[0,1],
                   [0,2],
                   [0,3],
                   [1,2],
                   [1,3],
                   [2,3]])
        
    for idx, file in enumerate(files):
        d = fits.open(file)
        h = d[0].header
        if idx == 0:
            if h['ESO INS SPEC RES'] == 'LOW':
                low = True
            else:
                low = False
            
        a1 = d['OI_VIS', 11].data['VISPHI']
        a2 = d['OI_VIS', 12].data['VISPHI']
        t = d['OI_VIS', 12].data['MJD'][::6]
        flag1 = d['OI_VIS', 11].data['FLAG']
        flag2 = d['OI_VIS', 12].data['FLAG']
        flag = flag1 + flag2
        a = (a1+a2)/2
        a[np.where(flag ==True)] = np.nan
        if low:
            aa = np.nanmean(a[:,2:-2],1)
        else:
            aa = np.nanmean(a[:,30:-30],1) 
        
        b = np.zeros((6, int(a.shape[0]/6)))
        b[0] = aa[::6]
        b[1] = aa[1::6]
        b[2] = aa[2::6]
        b[3] = aa[3::6]
        b[4] = aa[4::6]
        b[5] = aa[5::6]

        c = np.zeros_like(b)
        lst0 = h['LST']/3600
        time = d['OI_VIS', 11].data['TIME'][::6]/3600
        time -= time[0]
        lst = time + lst0
        angle = [get_angle_header_all(h, i, len(t)) for i in range(4)]
        wave = d['OI_WAVELENGTH',12].data['EFF_WAVE']*1e6
        for base in range(6):
            if bl_corr:
                if lst_corr:
                    if subspacing != 1:
                        lstdif = lst[-1] - lst[-2]
                        lst_s = np.linspace(lst[0], lst[-1] + lstdif, len(lst)*subspacing)
                        cor = -averaging(interp_list[base](lst_s), subspacing)[:-1]
                    else:
                        cor = -interp_list[base](lst)
                else:
                    t1 = bl_array[base][0]
                    t2 = bl_array[base][1]
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
                t1 = bl_array[base][0]
                t2 = bl_array[base][1]
                if lst_corr:
                    if subspacing != 1:
                        lstdif = lst[-1] - lst[-2]
                        lst_s = np.linspace(lst[0], lst[-1] + lstdif, len(lst)*subspacing)
                        cor1 = averaging(interp_list[t1](lst_s), subspacing)[:-1]
                        cor2 = averaging(interp_list[t2](lst_s), subspacing)[:-1]
                    else:
                        cor1 = interp_list[t1](lst)
                        cor2 = interp_list[t2](lst)  
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
                    
            wcor = np.zeros((len(t), len(wave)))
            for w in range(len(wave)):
                wcor[:,w] = cor/wave[w]*360
            c[base] -= np.mean(wcor[:,2:-2],1)

        if idx == 0:
            b0 = np.nanmean(b,1)
            c0 = np.nanmean(c,1)
        for bl in range(6):
            b[bl] -= b0[bl]
            c[bl] -= c0[bl]
            
        if idx == 0:
            visphi = b
            visphi_fake = c
            t_visphi = t
            ang = np.mean(np.asarray(angle),0)
            t_lst = lst
        else:
            visphi = np.concatenate((visphi,b),1)
            visphi_fake = np.concatenate((visphi_fake,c),1)
            t_visphi = np.concatenate((t_visphi,t))
            ang = np.concatenate((ang, np.mean(np.asarray(angle),0)))
            t_lst = np.concatenate((t_lst,lst))

    if plot:
        mjd_files = []
        ut_files = []
        lst_files = []
        for idx, file in enumerate(files):
            d = fits.open(file)
            mjd_files.append(d['OI_VIS', 12].data['MJD'][0])
            a = file.find('GRAVI.20')
            ut_files.append(file[a+17:a+22])
            lst_files.append(d[0].header['LST']/3600)
        tt = (t_visphi-t_visphi[0])*24*60
        t_files = (mjd_files-t_visphi[0])*24*60
        
        gs = gridspec.GridSpec(2,1, hspace=0.01)
        plt.figure(figsize=(8,10))
        off = 30

        ax = plt.subplot(gs[0,0])
        for idx in range(6):
            if lstplot:
                x = t_lst
            else:
                x = tt

            if idx == 0:
                plt.plot(averaging(x, plotav), averaging(visphi[idx], plotav)+idx*off, 
                         ls='-', lw=0.5, marker='o', zorder=10, color=colors_baseline[idx],
                         label='Data')
                plt.plot(averaging(x, plotav), averaging(visphi_fake[idx], plotav)+idx*off, 
                         ls='--', lw=0.5, marker='', zorder=10, color=colors_baseline[idx],
                         label='averaged TELFC_MCORR')
            else:
                plt.plot(averaging(x, plotav), averaging(visphi[idx], plotav)+idx*off, 
                         ls='-', lw=0.5, marker='o', zorder=10, color=colors_baseline[idx])
                plt.plot(averaging(x, plotav), averaging(visphi_fake[idx], plotav)+idx*off, 
                         ls='--', lw=0.5, marker='', zorder=10, color=colors_baseline[idx])
            blstd = np.nanstd(averaging(visphi_fake[idx], plotav) - averaging(visphi[idx], plotav))
            print('Bl %s std: %.2f' % (baseline[idx], blstd))
            plt.axhline(idx*off, lw=0.5, color=colors_baseline[idx])

        ang_av = averaging(ang, plotav)[:-1]
        ang_sep = int(len(ang_av)/len(t_files))
        if lstplot:
            for m in range(len(t_files)):
                plt.axvline(lst_files[m], ls='--', lw=0.5, color='grey')
                plt.text(lst_files[m]+0.01, -27, '%.1f$^\circ$' % (ang_av[::ang_sep][m]), rotation=90, fontsize=7)
        else:
            for m in range(len(t_files)):
                plt.axvline(t_files[m], ls='--', lw=0.5, color='grey')
                plt.text(t_files[m]+0.5, -27, '%.1f$^\circ$' % (ang_av[::ang_sep][m]), rotation=90, fontsize=7)
        plt.ylim(-35,185)
        plt.legend(loc=2)
        plt.ylabel('Phase [deg]')
        ax.set_xticklabels([])

        ax = plt.subplot(gs[1,0])
        for idx in range(6):
            plt.plot(averaging(x, plotav), averaging(visphi[idx], plotav)-averaging(visphi_fake[idx], plotav)+idx*off, 
                     ls='-', lw=0.5, marker='o', zorder=10, color=colors_baseline[idx],
                     label='Data')
            plt.axhline(idx*off, lw=0.5, color=colors_baseline[idx])
        if lstplot:
            for m in range(len(t_files)):
                plt.axvline(lst_files[m], ls='--', lw=0.5, color='grey')
                plt.text(lst_files[m]+0.01, 168, ut_files[m], rotation=90, fontsize=7)
            plt.xlabel('LST [h]')
        else:
            for m in range(len(t_files)):
                plt.axvline(t_files[m], ls='--', lw=0.5, color='grey')
                plt.text(t_files[m]+0.5, 168, ut_files[m], rotation=90, fontsize=7)
            plt.xlabel('Time [min]')
        plt.ylabel('Corr. Phase [deg]')
        plt.ylim(-35,185)
        plt.show()
                
    return t_visphi, t_lst, ang, visphi, visphi_fake



#####################
# load all corrections




#########################
# Correct a full night

class GravPhaseNight():
    def __init__(self, night, ndit, verbose=True, nopandas=False, pandasfile=None,
                 reddir=None, datadir='/data/user/forFrank2/'):
        """
        Package to do the full phase calibration, poscor, correction and fitting
        
        night       : night which shall be used
        ndit        : 1 to use the 5-min files or 32 for the 1frame reduction [1]
        
        """
        self.night = night
        self.ndit = ndit
        self.verbose = verbose
        
        nights = ['2019-03-27',
                '2019-03-28',
                '2019-03-31',
                '2019-04-15',
                '2019-04-16',
                '2019-04-18',
        #           '2019-04-19',
                '2019-04-21',
        #           '2019-06-14',
                '2019-06-16',
        #           '2019-06-17',
                '2019-06-19',
                '2019-06-20',
                '2019-07-15',
                '2019-07-17',
                '2019-08-13',
                '2019-08-14',
                '2019-08-15',
                '2019-08-18',
                '2019-08-19',
        #           '2019-09-11',
                '2019-09-12',
        #           '2019-09-13',
                '2019-09-15']
        calibrators = ['GRAVI.2019-03-28T08:00:22.802_dualscivis.fits',
                    'GRAVI.2019-03-29T07:35:36.327_dualscivis.fits',
                    'GRAVI.2019-04-01T06:53:20.843_dualscivis.fits',
                    'GRAVI.2019-04-16T08:08:45.675_dualscivis.fits',
                    'GRAVI.2019-04-17T09:08:30.918_dualscivis.fits',
                    'GRAVI.2019-04-19T06:00:32.198_dualscivis.fits',
        #                'GRAVI.2019-04-20T09:31:56.475_dualscivis.fits',
                    'GRAVI.2019-04-22T07:05:46.470_dualscivis.fits',
        #                'GRAVI.2019-06-15T04:40:02.407_dualscivis.fits',
                    'GRAVI.2019-06-17T04:44:03.992_dualscivis.fits',
        #                'GRAVI.2019-06-18T01:19:27.829_dualscivis.fits',
                    'GRAVI.2019-06-20T05:29:44.870_dualscivis.fits',
                    'GRAVI.2019-06-21T05:47:16.581_dualscivis.fits',
                    'GRAVI.2019-07-16T02:09:48.436_dualscivis.fits',
                    'GRAVI.2019-07-18T02:41:59.504_dualscivis.fits',
                    'GRAVI.2019-08-14T01:22:28.240_dualscivis.fits',
                    'GRAVI.2019-08-15T01:58:01.049_dualscivis.fits',
                    'GRAVI.2019-08-16T01:09:36.547_dualscivis.fits',
                    'GRAVI.2019-08-19T01:30:49.497_dualscivis.fits',#'GRAVI.2019-08-18T23:55:52.258_dualscivis.fits',
                    'GRAVI.2019-08-20T02:41:24.122_dualscivis.fits',
        #                'GRAVI.2019-09-12T01:26:51.547_dualscivis.fits',
                    'GRAVI.2019-09-12T23:48:18.886_dualscivis.fits',
        #                'GRAVI.2019-09-14T00:13:24.592_dualscivis.fits',
                    'GRAVI.2019-09-16T00:08:07.335_dualscivis.fits'
                    ] 
        try:
            self.calibrator = calibrators[nights.index(night)]
            if self.verbose:
                print('Night: %s, Calibrator: %s' % (night, self.calibrator))
        except ValueError:
            if self.verbose:
                print('Night is not available, try one of those:')
                print(nights)
            raise ValueError('Night is not available')
        
        if reddir is None:
            if ndit == 1:
                self.folder = datadir + night + '/reduced_PL20200513'
            elif ndit == 32:
                self.folder = datadir + night + '/reduced_PL20200513_1frame'
            else:
                raise ValueError('Ndit has to be 1 or 32')
        else:
            self.folder = datadir + night + '/' + reddir
        if self.verbose:
            print('Use data from: %s' % self.folder)
        self.bl_array = np.array([[0,1],
                            [0,2],
                            [0,3],
                            [1,2],
                            [1,3],
                            [2,3]])
        
        allfiles = sorted(glob.glob(self.folder + '/GRAVI*dualscivis.fits'))
        if len(allfiles) == 0:
            raise ValueError('No files found, most likely something is wrong with the reduction folder')
        
        sg_files = []
        s2_files = []
        for file in allfiles:
            h = fits.open(file)[0].header
            if h['ESO FT ROBJ NAME'] != 'IRS16C':
                continue
            if h['ESO INS SOBJ NAME'] == 'S2':
                if h['ESO INS SOBJ OFFX'] == 0:
                    s2_files.append(file)
                else:
                    sobjx = h['ESO INS SOBJ X']
                    sobjy = h['ESO INS SOBJ Y']
                    if -990 > sobjx or sobjx > -950:
                        continue
                    if -640 > sobjy or sobjy > -590:
                        continue
                    sg_files.append(file)
        if self.verbose:
            print('%i SGRA , %i S2 files found' % (len(sg_files), len(s2_files)))
        self.s2_files = s2_files
        self.sg_files = sg_files
        
        
        if not nopandas:
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
                pandasfile = resource_filename('gravipy', 'GRAVITY_DATA_2019_4_frame.object')
                pand = pd.read_pickle(pandasfile)
                
            sg_flux = []
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
                
                ddate = convert_date(obsdate)[0]
                orbitfile = resource_filename('gravipy', 's2_orbit_082020.txt')
                orbit = np.genfromtxt(orbitfile)
                s2_pos.append(orbit[find_nearest(orbit[:,0], ddate)][1:]*1000)
                    
            self.s2_pos = s2_pos
            self.sg_flux = sg_flux
            self.sg_header = sg_header

        
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
                      correction=True, poscor=True, plot=False,
                      fitstart=0, fitstop=-1, linear_cor=False,
                      save=False, ret=False, phasecorrection=None):
        """
        Function to corect, calibrate & poscor data
        mode        : mode for the correction. To see whats available run:
                      get_corrections()
        fluxcut     : if > 0 only uses the data which is brighter [0]
        subspacing  : spaces the applied corrections more than the data [1]
        correction  : apply correction [True]
        poscor      : apply poscor [True]
        plot        : Show some plots [False]
        linear_cor  : subtracts a polynom of 1st degree from each baseline [False]    
        fitstart    : file number from which to start the linear fit [0]
        fitstop     : file number where to stop measure the linear fit [-1]
        save        : saves the final data under one of the following folders [False]
                      (depending on the options)
                      /data/user/forFrank2/ + night + /reduced_PL20200513/correction_ + mode + _new/
                      /data/user/forFrank2/ + night + /reduced_PL20200513_1frame/correction_ + mode + _new/ 
        ret         : Returns list with all results
        """
        ndit = self.ndit

        s2_files = self.s2_files
        sg_files = self.sg_files
        if fluxcut > 0:
            sg_flux = self.sg_flux
        
        if isinstance(mode, str):
            corrections_dict = self.get_corrections(bequiet=True)
            try:
                interp_list = corrections_dict[mode]
            except KeyError:
                print('mode not avialable, use one of those:')
                print(corrections_dict.keys())
                print('More corrections in: /data/user/fwidmann/Phase_fit_cor/corrections_data/')
                raise KeyError

            if 'lst' in mode:
                lst_corr = True
                if ndit == 1:
                    raise ValueError('Ndit 1 and lst correction is not implemented properly')
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
                raise ValueError('If interpolations are given mode has to be a list of four')
            
            
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
            
            
        ################
        # read in al necessary data
        ################
        
        sg_visphi_p1 = np.zeros((len(sg_files), ndit*6, 14))*np.nan
        sg_visphi_p2 = np.zeros((len(sg_files), ndit*6, 14))*np.nan
        sg_visphi_err_p1 = np.zeros((len(sg_files), ndit*6, 14))*np.nan
        sg_visphi_err_p2 = np.zeros((len(sg_files), ndit*6, 14))*np.nan
        sg_t = np.zeros((len(sg_files), ndit))*np.nan
        sg_lst = np.zeros((len(sg_files), ndit))*np.nan
        sg_ang = np.zeros((len(sg_files), ndit))*np.nan
        sg_u = np.zeros((len(sg_files), ndit*6, 14))*np.nan
        sg_v = np.zeros((len(sg_files), ndit*6, 14))*np.nan
        sg_u_raw = np.zeros((len(sg_files), ndit*6))*np.nan
        sg_v_raw = np.zeros((len(sg_files), ndit*6))*np.nan
        sg_correction = np.zeros((len(sg_files), ndit*6, 14))*np.nan
        sg_correction_wo = np.zeros((len(sg_files), ndit*6, 14))*np.nan
        
        for fdx, file in enumerate(sg_files):
            d = fits.open(file)
            h = d[0].header
            if fluxcut > 0:
                if f < self.sg_flux[fdx]:
                    continue
            sg_t[fdx] = d['OI_VIS', 12].data['MJD'][::6]
            o_ndit = h['ESO DET2 NDIT']
            lst0 = h['LST']/3600
            time = d['OI_VIS', 11].data['TIME'][::6]/3600
            time -= time[0]
            sg_lst[fdx] = time + lst0

            U = d['OI_VIS',11].data['UCOORD']
            V = d['OI_VIS',11].data['VCOORD']
            sg_u_raw[fdx] = U
            sg_v_raw[fdx] = V
            wave = d['OI_WAVELENGTH',11].data['EFF_WAVE']
            U_as = np.zeros((ndit*6,14))
            V_as = np.zeros((ndit*6,14))
            for bl in range(ndit*6):
                for wdx, wl in enumerate(wave):
                    U_as[bl,wdx] = U[bl]/wl * np.pi / 180. / 3600./1000
                    V_as[bl,wdx] = V[bl]/wl * np.pi / 180. / 3600./1000
            sg_u[fdx] = U_as
            sg_v[fdx] = V_as

            angle = [get_angle_header_all(h, i, o_ndit) for i in range(4)]
            sg_ang[fdx] = np.mean([get_angle_header_all(h, i, ndit) for i in range(4)], 0)
            wave = d['OI_WAVELENGTH',12].data['EFF_WAVE']*1e6
            dlambda = d['OI_WAVELENGTH',11].data['EFF_BAND']/2*1e6
            fullcor = np.zeros((6*ndit, len(wave)))

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
                        wcor[:,w] = cor/wave[w]*360
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
            sg_correction[fdx] = fullcor

            flag1 = d['OI_VIS', 11].data['FLAG']
            flag2 = d['OI_VIS', 12].data['FLAG']

            if correction:
                sg_visphi_p1[fdx] = d['OI_VIS', 11].data['VISPHI'] + fullcor
                sg_visphi_p2[fdx] = d['OI_VIS', 12].data['VISPHI'] + fullcor
            else:
                sg_visphi_p1[fdx] = d['OI_VIS', 11].data['VISPHI']
                sg_visphi_p2[fdx] = d['OI_VIS', 12].data['VISPHI']

            sg_visphi_err_p1[fdx] = d['OI_VIS', 11].data['VISPHIERR']
            sg_visphi_err_p2[fdx] = d['OI_VIS', 12].data['VISPHIERR']

            sg_visphi_p1[fdx][np.where(flag1 == True)] = np.nan
            sg_visphi_p2[fdx][np.where(flag2 == True)] = np.nan
            sg_visphi_err_p1[fdx][np.where(flag1 == True)] = np.nan
            sg_visphi_err_p2[fdx][np.where(flag2 == True)] = np.nan

        self.sg_u_raw = sg_u_raw
        self.sg_v_raw = sg_v_raw
        self.wave = wave
        self.dlambda = dlambda

        s2_visphi_p1 = np.zeros((len(s2_files), ndit*6, 14))
        s2_visphi_p2 = np.zeros((len(s2_files), ndit*6, 14))
        s2_visphi_err_p1 = np.zeros((len(s2_files), ndit*6, 14))
        s2_visphi_err_p2 = np.zeros((len(s2_files), ndit*6, 14))
        s2_t = np.zeros((len(s2_files), ndit))
        s2_lst = np.zeros((len(s2_files), ndit))
        s2_ang = np.zeros((len(s2_files), ndit))
        s2_u = np.zeros((len(s2_files), ndit*6, 14))
        s2_v = np.zeros((len(s2_files), ndit*6, 14))
        s2_u_raw = np.zeros((len(s2_files), ndit*6))
        s2_v_raw = np.zeros((len(s2_files), ndit*6))
        s2_correction = np.zeros((len(s2_files), ndit*6, 14))*np.nan
        s2_correction_wo = np.zeros((len(s2_files), ndit*6, 14))*np.nan

        for fdx, file in enumerate(s2_files):
            d = fits.open(file)
            h = d[0].header
            s2_t[fdx] = d['OI_VIS', 12].data['MJD'][::6]

            o_ndit = h['ESO DET2 NDIT']
            lst0 = h['LST']/3600
            time = d['OI_VIS', 11].data['TIME'][::6]/3600
            time -= time[0]
            s2_lst[fdx] = time + lst0

            U = d['OI_VIS',11].data['UCOORD']
            V = d['OI_VIS',11].data['VCOORD']
            s2_u_raw[fdx] = U
            s2_v_raw[fdx] = V
            wave = d['OI_WAVELENGTH',11].data['EFF_WAVE']
            U_as = np.zeros((ndit*6,14))
            V_as = np.zeros((ndit*6,14))
            for bl in range(ndit*6):
                for wdx, wl in enumerate(wave):
                    U_as[bl,wdx] = U[bl]/wl * np.pi / 180. / 3600./1000
                    V_as[bl,wdx] = V[bl]/wl * np.pi / 180. / 3600./1000
            s2_u[fdx] = U_as
            s2_v[fdx] = V_as

            angle = [get_angle_header_all(h, i, o_ndit) for i in range(4)]
            s2_ang[fdx] = np.mean([get_angle_header_all(h, i, ndit) for i in range(4)], 0)
            wave = d['OI_WAVELENGTH',12].data['EFF_WAVE']*1e6
            fullcor = np.zeros((6*ndit, len(wave)))
            
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
                        wcor[:,w] = cor/wave[w]*360
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
            s2_correction[fdx] = fullcor

            flag1 = d['OI_VIS', 11].data['FLAG']
            flag2 = d['OI_VIS', 12].data['FLAG']
            

            if correction:
                s2_visphi_p1[fdx] = d['OI_VIS', 11].data['VISPHI'] + fullcor
                s2_visphi_p2[fdx] = d['OI_VIS', 12].data['VISPHI'] + fullcor
            else:
                s2_visphi_p1[fdx] = d['OI_VIS', 11].data['VISPHI']
                s2_visphi_p2[fdx] = d['OI_VIS', 12].data['VISPHI']

            s2_visphi_err_p1[fdx] = d['OI_VIS', 11].data['VISPHIERR']
            s2_visphi_err_p2[fdx] = d['OI_VIS', 12].data['VISPHIERR']

            s2_visphi_p1[fdx][np.where(flag1 ==True)] = np.nan
            s2_visphi_p2[fdx][np.where(flag2 ==True)] = np.nan
            s2_visphi_err_p1[fdx][np.where(flag1 ==True)] = np.nan
            s2_visphi_err_p2[fdx][np.where(flag2 ==True)] = np.nan

        self.s2_u_raw = s2_u_raw
        self.s2_v_raw = s2_u_raw
        
        tstart = np.nanmin((np.nanmin(s2_t), np.nanmin(sg_t)))
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

        mc_p1 = np.zeros((ndit, 6, 14))
        mc_p2 = np.zeros((ndit, 6, 14))
        cal_p1 = np.zeros_like(c_p1, dtype=np.complex)
        cal_p2 = np.zeros_like(c_p2, dtype=np.complex)

        for bl in range(6):
            mc_p1[:,bl,:] = c_p1[bl::6,:]
            mc_p2[:,bl,:] = c_p2[bl::6,:]


        mc_p1 = np.exp(1j*mc_p1/180*np.pi)
        mc_p1 = np.nanmean(mc_p1,0)
        mc_p2 = np.exp(1j*mc_p2/180*np.pi)
        mc_p2 = np.nanmean(mc_p2,0)

        for dit in range(ndit):
            cal_p1[dit*6:(dit+1)*6] = mc_p1
            cal_p2[dit*6:(dit+1)*6] = mc_p2

        for fdx, file in enumerate(sg_files):
            sg_visphi_p1[fdx] = np.angle((np.exp(1j*sg_visphi_p1[fdx]/180*np.pi) / cal_p1), deg=True)
            sg_visphi_p2[fdx] = np.angle((np.exp(1j*sg_visphi_p2[fdx]/180*np.pi) / cal_p2), deg=True)

        for fdx, file in enumerate(s2_files):
            s2_visphi_p1[fdx] = np.angle((np.exp(1j*s2_visphi_p1[fdx]/180*np.pi) / cal_p1), deg=True)
            s2_visphi_p2[fdx] = np.angle((np.exp(1j*s2_visphi_p2[fdx]/180*np.pi) / cal_p2), deg=True)

        ##########################
        # Poscor
        ##########################

        if poscor:
            s2_B = np.zeros((s2_u.shape[0], 6, 14, 2))
            SpFreq = np.zeros((s2_u.shape[0], 6, 14))
            for bl in range(6):
                if ndit == 1:
                    s2_B[:,bl,:,0] = s2_u[:,bl,:]
                    s2_B[:,bl,:,1] = s2_v[:,bl,:]
                    SpFreq[:,bl,:] = np.sqrt(s2_u[:,bl,:]**2 + s2_v[:,bl,:]**2)
                else:
                    s2_u[s2_u==0] = np.nan
                    s2_v[s2_v==0] = np.nan
                    s2_B[:,bl,:,0] = np.nanmean(s2_u[:,bl::6,:],1)
                    s2_B[:,bl,:,1] = np.nanmean(s2_v[:,bl::6,:],1)
                    SpFreq[:,bl,:] = np.sqrt(np.nanmean(s2_u[:,bl::6,:],1)**2 + np.nanmean(s2_v[:,bl::6,:],1)**2)

            s2_dB = np.copy(s2_B)
            s2_dB = s2_dB - s2_B[calib_file]

            dB1 = np.transpose([s2_dB[:,:,:,0].flatten(),s2_dB[:,:,:,1].flatten()])

            nfiles = len(s2_visphi_p1)
            s2_visphi_fit = np.zeros((nfiles, 6, 14))
            s2_visphi_err_fit = np.zeros((nfiles, 6, 14))
            if ndit == 1:
                for bl in range(6):
                    s2_visphi_fit[:,bl,:] = s2_visphi_p1[:,bl,:]
                    s2_visphi_fit[:,bl,:] += s2_visphi_p2[:,bl,:]
                    s2_visphi_err_fit[:,bl,:] = s2_visphi_err_p1[:,bl,:]
                    s2_visphi_err_fit[:,bl,:] += s2_visphi_err_p2[:,bl,:]
            else:
                for bl in range(6):
                    s2_visphi_fit[:,bl,:] = np.nanmean(s2_visphi_p1[:,bl::6,:],1)
                    s2_visphi_fit[:,bl,:] += np.nanmean(s2_visphi_p2[:,bl::6,:],1)
                    s2_visphi_err_fit[:,bl,:] = np.sqrt(np.nanmean(s2_visphi_err_p1[:,bl::6,:],1)**2+
                                                np.nanmean(s2_visphi_err_p2[:,bl::6,:],1)**2)/np.sqrt(2)

            s2_visphi_fit /= 2
            s2_visphi_fit[:,:,:2] = np.nan
            s2_visphi_fit[:,:,-2:] = np.nan

            Vphi_err = s2_visphi_err_fit.flatten()
            Vphi = s2_visphi_fit.flatten()

            Vphi2 = Vphi[~np.isnan(Vphi)]/360
            Vphi_err2 = Vphi_err[~np.isnan(Vphi)]/360
            dB2 = np.zeros((len(Vphi2),2))
            dB2[:,0] = dB1[~np.isnan(Vphi),0]
            dB2[:,1] = dB1[~np.isnan(Vphi),1]

            def f(dS):
                Chi = (Vphi2-np.dot(dB2,dS))/(Vphi_err2)
                return Chi

            try:
                dS, pcov, infodict, errmsg, success = optimize.leastsq(f, x0=[0,0], full_output=1)
            except TypeError:
                print('PosCor failed')
                dS = [0,0]

            if self.verbose:
                print('Applied poscor: (%.3f,%.3f) mas ' % (dS[0], dS[1]))
                print('Chi2 of poscor: %.2f \n' % np.sum(f(dS)**2))

            if plot:
                n = nfiles
                colors = plt.cm.inferno(np.linspace(0,1,n+2))
                fitres = np.dot(dB1,dS)*360
                fitres_r = np.reshape(fitres, (n, 6, len(SpFreq[0, 0])))
                for idx in range(n):
                    for bl in range(6):
                        plt.errorbar(SpFreq[idx, bl,2:-2]*1000, s2_visphi_fit[idx, bl,2:-2], 
                                     s2_visphi_err_fit[idx, bl,2:-2], 
                                     ls='', marker='o', color=colors[idx], alpha=0.5)
                        plt.plot(SpFreq[idx,bl]*1000, fitres_r[idx,bl], color=colors[idx])
                plt.ylim(-100, 100)
                plt.xlabel('Spatial frequency [1/as]')
                plt.ylabel('Visibility phase [deg]')
                plt.title('Poscor')
                plt.show()

            s2_B = np.zeros((s2_u.shape[0], ndit*6, 14, 2))
            s2_B[:,:,:,0] = s2_u
            s2_B[:,:,:,1] = s2_v
            s2_dB = np.copy(s2_B)

            B_calib = np.zeros((6, 14, 2))
            for bl in range(6):
                B_calib[bl] = np.nanmean(s2_B[calib_file][bl::6],0)
                s2_dB[:,bl::6,:,:] = s2_dB[:,bl::6,:,:] - B_calib[bl]

            sg_B = np.zeros((sg_u.shape[0], ndit*6, 14, 2))
            sg_B[:,:,:,0] = sg_u
            sg_B[:,:,:,1] = sg_v
            sg_dB = np.copy(sg_B)

            B_calib = np.zeros((6, 14, 2))
            for bl in range(6):
                B_calib[bl] = np.nanmean(s2_B[calib_file][bl::6],0)
                sg_dB[:,bl::6,:,:] = sg_dB[:,bl::6,:,:] - B_calib[bl]

            sg_visphi_p1 -= np.dot(sg_dB,dS)*360
            sg_visphi_p2 -= np.dot(sg_dB,dS)*360

            s2_visphi_p1 -= np.dot(s2_dB,dS)*360
            s2_visphi_p2 -= np.dot(s2_dB,dS)*360
            
        
        ##########################
        # linear bl fit
        ##########################

        if linear_cor:
            def linreg(x, a, b):
                return a*x + b

            for bl in range(6):
                x = sg_t[fitstart:fitstop].flatten()
                y = np.nanmean(sg_visphi_p1[fitstart:fitstop,bl::6,2:-2],2).flatten()
                yerr = np.mean(sg_visphi_err_p1[fitstart:fitstop,bl::6,2:-2],2).flatten()
                valid = ~(np.isnan(x) | np.isnan(y))
                popt, pcov = optimize.curve_fit(linreg, x[valid], y[valid], sigma=yerr[valid], p0=[0,0])
                for wdx in range(len(wave)):
                    sg_visphi_p1[:,bl::6,wdx] -= linreg(sg_t, *popt)

                y = np.nanmean(sg_visphi_p2[fitstart:fitstop,bl::6,2:-2],2).flatten()
                yerr = np.mean(sg_visphi_err_p2[fitstart:fitstop,bl::6,2:-2],2).flatten()
                valid = ~(np.isnan(x) | np.isnan(y))
                popt, pcov = optimize.curve_fit(linreg, x[valid], y[valid], sigma=yerr[valid], p0=[0,0])
                for wdx in range(len(wave)):
                    sg_visphi_p2[:,bl::6,wdx] -= linreg(sg_t, *popt)
                    
                    
        
        ##########################
        # unwrap phases
        ##########################
        
        sg_visphi_p1 = ((sg_visphi_p1+180)%360)-180
        sg_visphi_p2 = ((sg_visphi_p2+180)%360)-180
        s2_visphi_p1 = ((s2_visphi_p1+180)%360)-180
        s2_visphi_p2 = ((s2_visphi_p2+180)%360)-180
        
        

        result = [[sg_t, sg_lst, sg_ang, sg_visphi_p1, sg_visphi_err_p1, sg_visphi_p2, sg_visphi_err_p2],
                  [s2_t, s2_lst, s2_ang, s2_visphi_p1, s2_visphi_err_p1, s2_visphi_p2, s2_visphi_err_p2]]

        
        if plot:
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
                if linear_cor:
                    folder = file[:file.find('GRAVI')] + 'correction_' + mode +'_new/'
                else:
                    folder = file[:file.find('GRAVI')] + 'correction_' + mode +'_nolin/'
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
            else:
                if linear_cor:
                    folder = file[:file.find('GRAVI')] + 'correction_nocor_new/'
                else:
                    folder = file[:file.find('GRAVI')] + 'correction_nocor_nolin/'
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
                        
        self.alldata = result
        if ret:
            return self.alldata
        
        


    #########################
    # Fitting functions

    def pointsource(self,uv, wave, x, y, mask=1, flatten=True):
        mas2rad = 1e-3 / 3600 / 180 * np.pi
        u = uv[0]
        v = uv[1]
        x_uas = x*mas2rad*1e6
        y_uas = y*mas2rad*1e6
        phi = np.zeros((6, len(wave)))
        for wdx in range(len(wave)):
            phi[:,wdx] = (u*x_uas + v*y_uas)/wave[wdx]
        visphi = np.angle(np.exp(-2*np.pi*1j*phi), deg=True)
        if flatten:
            return visphi.flatten()[mask]
        else:
            return visphi

    def fit_pointsource(self,u,v,wave,visphi,visphierr,plot=False):
        uv = [u.flatten(),v.flatten()]
        visphif = visphi.flatten()
        visphierrf = visphierr.flatten()
        mask = ~np.isnan(visphif) * ~np.isnan(visphierrf)
        popt, pcov = optimize.curve_fit(lambda uv, x, y: self.pointsource(uv, wave, x, y, mask),
                                        uv, visphif[mask], sigma=visphierrf[mask],
                                        bounds=(-10,10))
        
        if plot:
            rad2as = 180 / np.pi * 3600
            wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
            u_as = np.zeros((len(u),len(wave)))
            v_as = np.zeros((len(v),len(wave))) 
            u_as_model = np.zeros((len(u),len(wave_model)))
            v_as_model = np.zeros((len(v),len(wave_model)))
            for i in range(0,len(u)):
                u_as[i,:] = u[i]/(wave*1.e-6) / rad2as
                v_as[i,:] = v[i]/(wave*1.e-6) / rad2as
                u_as_model[i,:] = u[i]/(wave_model*1.e-6) / rad2as
                v_as_model[i,:] = v[i]/(wave_model*1.e-6) / rad2as
            magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)
            magu_as = np.sqrt(u_as**2.+v_as**2.)   
            
            model_visphi = self.pointsource(uv, wave_model, popt[0], popt[1], mask=1, flatten=False)
            
            for i in range(0,6):
                plt.errorbar(magu_as[i,:], visphi[i,:], visphierr[i,:], 
                            label=baseline[i], color=colors_baseline[i], 
                            ls='', lw=1, alpha=0.5, capsize=0)
                plt.scatter(magu_as[i,:], visphi[i,:],
                            color=colors_baseline[i], alpha=0.5)
                plt.plot(magu_as_model[i,:], model_visphi[i,:],
                        color='k', zorder=100)
            plt.ylabel('visibility phase [deg]')
            plt.xlabel('spatial frequency [1/arcsec]')
            plt.show()
        return popt



        
        
        
        
    def fit_night_1src(self, plot=True, plotfits=False, ret_flux=True, only_sgr=False, fitcut=2):
        """
        Fit a pointsource model to all data from the night
        """
        ndit = self.ndit
        wave = self.wave
        sg_files = self.sg_files
        s2_files = self.s2_files
        sg_u_raw = self.sg_u_raw
        sg_v_raw = self.sg_v_raw
        s2_u_raw = self.s2_u_raw
        s2_v_raw = self.s2_v_raw

        [[sg_t, sg_lst, sg_ang, sg_visphi_p1, sg_visphi_err_p1, sg_visphi_p2, sg_visphi_err_p2],
         [s2_t, s2_lst, s2_ang, s2_visphi_p1, s2_visphi_err_p1, s2_visphi_p2, s2_visphi_err_p2]] = self.alldata

        if not only_sgr:
            s2_ra_p1 = np.zeros((len(s2_files), ndit))*np.nan
            s2_de_p1 = np.zeros((len(s2_files), ndit))*np.nan
            s2_ra_p2 = np.zeros((len(s2_files), ndit))*np.nan
            s2_de_p2 = np.zeros((len(s2_files), ndit))*np.nan
            for fdx, file in enumerate(s2_files):
                for dit in range(ndit):
                    u = s2_u_raw[fdx, dit*6:(dit+1)*6]
                    v = s2_v_raw[fdx, dit*6:(dit+1)*6]
                    if np.sum(u==0) > 0:
                        continue

                    visphi = s2_visphi_p1[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                    visphierr = s2_visphi_err_p1[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                    wcut = np.copy(wave)[fitcut:-fitcut]
                    if np.sum(np.isnan(visphi)) > 10:
                        continue
                    s2_ra_p1[fdx, dit], s2_de_p1[fdx, dit] = self.fit_pointsource(u,v,wcut,visphi,visphierr,plot=plotfits)

                    visphi = s2_visphi_p2[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                    visphierr = s2_visphi_err_p2[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                    wcut = np.copy(wave)[fitcut:-fitcut]
                    if np.sum(np.isnan(visphi)) > 10:
                        continue
                    s2_ra_p2[fdx, dit], s2_de_p2[fdx, dit] = self.fit_pointsource(u,v,wcut,visphi,visphierr,plot=plotfits)

        sg_ra_p1 = np.zeros((len(sg_files), ndit))*np.nan
        sg_de_p1 = np.zeros((len(sg_files), ndit))*np.nan
        sg_ra_p2 = np.zeros((len(sg_files), ndit))*np.nan
        sg_de_p2 = np.zeros((len(sg_files), ndit))*np.nan

        sg_flux = self.sg_flux
        for fdx, file in enumerate(sg_files):
            for dit in range(ndit):
                u = sg_u_raw[fdx, dit*6:(dit+1)*6]
                v = sg_v_raw[fdx, dit*6:(dit+1)*6]
                if np.sum(u==0) > 0:
                    continue

                visphi = sg_visphi_p1[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                visphierr = sg_visphi_err_p1[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                wcut = np.copy(wave)[fitcut:-fitcut]
                if np.sum(np.isnan(visphi)) > 10:
                    continue
                sg_ra_p1[fdx, dit], sg_de_p1[fdx, dit] = self.fit_pointsource(u,v,wcut,visphi,visphierr,plot=plotfits)

                visphi = sg_visphi_p2[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                visphierr = sg_visphi_err_p2[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                wcut = np.copy(wave)[fitcut:-fitcut]
                if np.sum(np.isnan(visphi)) > 10:
                    continue
                sg_ra_p2[fdx, dit], sg_de_p2[fdx, dit] = self.fit_pointsource(u,v,wcut,visphi,visphierr,plot=plotfits)
                
            
        if plot:
            if ndit == 1:
                umark = 'o'
            else:
                umark = '.'
            plt.figure(figsize=(7,6))
            gs = gridspec.GridSpec(3,1, hspace=0.05)
            axis = plt.subplot(gs[0,0])
            plt.title('Position fit')
            plt.plot(sg_t.flatten()[::ndit], sg_flux, 
                     color=color1, ls='-',lw=0.5, marker='o')
            axis.set_xticklabels([])
            plt.ylabel('Flux [S2]')
            plt.ylim(0,1.5)
            axis = plt.subplot(gs[1,0])
            if not only_sgr:
                plt.plot(s2_t.flatten(), (s2_ra_p1.flatten()+s2_ra_p2.flatten())/2, 
                     color='k', ls='', marker=umark, label='S2')
            plt.plot(sg_t.flatten(), (sg_ra_p1.flatten()+sg_ra_p2.flatten())/2, 
                     color=color1, ls='', marker=umark, label='SgrA*')
            plt.axhline(0, color='grey', lw=0.5, zorder=0)
            plt.legend()
            axis.set_xticklabels([])
            plt.ylabel('RA [mas]')
            axis = plt.subplot(gs[2,0])
            if not only_sgr:
                plt.plot(s2_t.flatten(), (s2_de_p1.flatten()+s2_de_p2.flatten())/2, 
                        color='k', ls='', marker=umark)
            plt.plot(sg_t.flatten(), (sg_de_p1.flatten()+sg_de_p2.flatten())/2, 
                     color=color1, ls='', marker=umark)
            plt.axhline(0, color='grey', lw=0.5, zorder=0)
            plt.xlabel('Time [min]')
            plt.ylabel('Dec [mas]')
            plt.show()
        
        if not only_sgr:
            fitres = [[sg_t, sg_ra_p1, sg_de_p1, sg_ra_p2, sg_de_p2],
                      [s2_t, s2_ra_p1, s2_de_p1, s2_ra_p2, s2_de_p2]]
        else:
            fitres = [sg_t, sg_ra_p1, sg_de_p1, sg_ra_p2, sg_de_p2]
        if ret_flux:
            return sg_flux, fitres
        else:
            return fitres
        
        
        
    def fit_night_1src_mcore(self, nthreads, ret_flux=True, fitcut=2):
        """
        Fit a pointsource model to all data from the night
        """
        ndit = self.ndit
        wave = self.wave
        sg_files = self.sg_files
        s2_files = self.s2_files
        sg_u_raw = self.sg_u_raw
        sg_v_raw = self.sg_v_raw
        s2_u_raw = self.s2_u_raw
        s2_v_raw = self.s2_v_raw

        [[sg_t, sg_lst, sg_ang, sg_visphi_p1, sg_visphi_err_p1, sg_visphi_p2, sg_visphi_err_p2],
         [s2_t, s2_lst, s2_ang, s2_visphi_p1, s2_visphi_err_p1, s2_visphi_p2, s2_visphi_err_p2]] = self.alldata

        sg_ra_p1 = np.zeros((len(sg_files), ndit))*np.nan
        sg_de_p1 = np.zeros((len(sg_files), ndit))*np.nan
        sg_ra_p2 = np.zeros((len(sg_files), ndit))*np.nan
        sg_de_p2 = np.zeros((len(sg_files), ndit))*np.nan

        sg_flux = self.sg_flux
        
        
        
        #Parallel(n_jobs=args.numcores)(delayed(fit_file)(f, t) for t, f in enumerate(files))
        #pool = multiprocessing.Pool(args.numcores)
        #pool.map(fit_file, files)
        
        def _fitfile(fdx):
            sg_ra_p1_file = np.zeros(ndit)*np.nan
            sg_de_p1_file = np.zeros(ndit)*np.nan
            sg_ra_p2_file = np.zeros(ndit)*np.nan
            sg_de_p2_file = np.zeros(ndit)*np.nan
            
            for dit in range(ndit):
                u = sg_u_raw[fdx, dit*6:(dit+1)*6]
                v = sg_v_raw[fdx, dit*6:(dit+1)*6]
                if np.sum(u==0) > 0:
                    continue

                visphi = sg_visphi_p1[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                visphierr = sg_visphi_err_p1[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                wcut = np.copy(wave)[fitcut:-fitcut]
                if np.sum(np.isnan(visphi)) > 10:
                    continue
                sg_ra_p1_file[dit], sg_de_p1_file[dit] = self.fit_pointsource(u,v,wcut,visphi,visphierr,plot=False)

                visphi = sg_visphi_p2[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                visphierr = sg_visphi_err_p2[fdx, dit*6:(dit+1)*6][:,fitcut:-fitcut]
                wcut = np.copy(wave)[fitcut:-fitcut]
                if np.sum(np.isnan(visphi)) > 10:
                    continue
                sg_ra_p2_file[dit], sg_de_p2_file[dit] = self.fit_pointsource(u,v,wcut,visphi,visphierr,plot=False)  
            return sg_ra_p1_file, sg_de_p1_file, sg_ra_p2_file, sg_de_p2_file
        
        filelen = np.arange(len(sg_files))
        res = Parallel(n_jobs=nthreads)(delayed(_fitfile)(f) for f in filelen)
        #pool = multiprocessing.Pool(nthreads)
        #res = pool.map(_fitfile, filelen)
        
        return res
        
        
        #for fdx, file in enumerate(sg_files):
            
        #if plot:
            #if ndit == 1:
                #umark = 'o'
            #else:
                #umark = '.'
            #plt.figure(figsize=(7,6))
            #gs = gridspec.GridSpec(3,1, hspace=0.05)
            #axis = plt.subplot(gs[0,0])
            #plt.title('Position fit')
            #plt.plot(sg_t.flatten()[::ndit], sg_flux, 
                     #color=color1, ls='-',lw=0.5, marker='o')
            #axis.set_xticklabels([])
            #plt.ylabel('Flux [S2]')
            #plt.ylim(0,1.5)
            #axis = plt.subplot(gs[1,0])
            #plt.plot(sg_t.flatten(), (sg_ra_p1.flatten()+sg_ra_p2.flatten())/2, 
                     #color=color1, ls='', marker=umark, label='SgrA*')
            #plt.axhline(0, color='grey', lw=0.5, zorder=0)
            #plt.legend()
            #axis.set_xticklabels([])
            #plt.ylabel('RA [mas]')
            #axis = plt.subplot(gs[2,0])
            #plt.plot(sg_t.flatten(), (sg_de_p1.flatten()+sg_de_p2.flatten())/2, 
                     #color=color1, ls='', marker=umark)
            #plt.axhline(0, color='grey', lw=0.5, zorder=0)
            #plt.xlabel('Time [min]')
            #plt.ylabel('Dec [mas]')
            #plt.show()
        
        #fitres = [sg_t, sg_ra_p1, sg_de_p1, sg_ra_p2, sg_de_p2]
        #if ret_flux:
            #return sg_flux, fitres
        #else:
            #return fitres
        
        
            
            
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


    def threesource(self, uv, wave, dlambda, sources, x, y, mask=1,
                    alpha_SgrA=-0.5,alpha_S=3, alpha_bg=3, 
                    fluxRatioBG=0, ret_flat=True):
        phasemaps = self.fit_phasemaps
        mas2rad = 1e-3 / 3600 / 180 * np.pi
        u = uv[0]
        v = uv[1]
        
        if phasemaps:
            s2_pos, s2_fr, f1_pos, f1_fr, cor = sources
            cor_amp_s2, cor_pha_s2, cor_amp_f1, cor_pha_f1 = cor
            
            
            pm_amp_f1 = np.array([[cor_amp_f1[0], cor_amp_f1[1]],
                                    [cor_amp_f1[0], cor_amp_f1[2]],
                                    [cor_amp_f1[0], cor_amp_f1[3]],
                                    [cor_amp_f1[1], cor_amp_f1[2]],
                                    [cor_amp_f1[1], cor_amp_f1[3]],
                                    [cor_amp_f1[2], cor_amp_f1[3]]])
            pm_pha_f1 = np.array([[cor_pha_f1[0], cor_pha_f1[1]],
                                    [cor_pha_f1[0], cor_pha_f1[2]],
                                    [cor_pha_f1[0], cor_pha_f1[3]],
                                    [cor_pha_f1[1], cor_pha_f1[2]],
                                    [cor_pha_f1[1], cor_pha_f1[3]],
                                    [cor_pha_f1[2], cor_pha_f1[3]]])
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
            vis = np.zeros((6,len(wave))) + 0j
            for i in range(0,6):
                s_SgrA = ((x)*u[i] + (y)*v[i]) * mas2rad * 1e6
                s_S2 = ((s2_pos[0]+x)*u[i] + (s2_pos[1]+y)*v[i]) * mas2rad * 1e6
                s_F1 = ((f1_pos[0]+x)*u[i] + (f1_pos[1]+y)*v[i]) * mas2rad * 1e6
                
                opd_f1 = (pm_pha_f1[i,0] - pm_pha_f1[i,1])/360*wave
                opd_s2 = (pm_pha_s2[i,0] - pm_pha_s2[i,1])/360*wave
                s_F1 -= opd_f1
                s_S2 -= opd_s2
                
                cr1_s2 = pm_amp_s2[i,0]
                cr2_s2 = pm_amp_s2[i,1]
                cr1_f1 = pm_amp_f1[i,0]
                cr2_f1 = pm_amp_f1[i,1]
                
                intSgrA = self.vis_intensity_approx(s_SgrA, alpha_SgrA, wave, dlambda)
                intSgrA_center = self.vis_intensity_approx(0, alpha_SgrA, wave, dlambda)
                intS2 = self.vis_intensity_approx(s_S2, alpha_S, wave, dlambda)
                intS2_center = self.vis_intensity_approx(0, alpha_S, wave, dlambda)
                intF1 = self.vis_intensity_approx(s_F1, alpha_S, wave, dlambda)
                intF1_center = self.vis_intensity_approx(0, alpha_S, wave, dlambda)
                intBG = self.vis_intensity_approx(0, alpha_bg, wave, dlambda)

                vis[i,:] = ((intSgrA + cr1_f1*cr2_f1*f1_fr*intF1 + cr1_s2*cr2_s2*s2_fr*intS2)/
                            (intSgrA_center + cr1_f1*cr2_f1*f1_fr*intF1_center + 
                            cr1_s2*cr2_s2*s2_fr*intS2_center + fluxRatioBG*intBG))
            
        else:
            s2_pos, s2_fr, f1_pos, f1_fr = sources

            vis = np.zeros((6,len(wave))) + 0j
            for i in range(0,6):
                s_SgrA = ((x)*u[i] + (y)*v[i]) * mas2rad * 1e6
                s_S2 = ((s2_pos[0]+x)*u[i] + (s2_pos[1]+y)*v[i]) * mas2rad * 1e6
                s_F1 = ((f1_pos[0]+x)*u[i] + (f1_pos[1]+y)*v[i]) * mas2rad * 1e6
                
                intSgrA = self.vis_intensity_approx(s_SgrA, alpha_SgrA, wave, dlambda)
                intSgrA_center = self.vis_intensity_approx(0, alpha_SgrA, wave, dlambda)
                intS2 = self.vis_intensity_approx(s_S2, alpha_S, wave, dlambda)
                intS2_center = self.vis_intensity_approx(0, alpha_S, wave, dlambda)
                intF1 = self.vis_intensity_approx(s_F1, alpha_S, wave, dlambda)
                intF1_center = self.vis_intensity_approx(0, alpha_S, wave, dlambda)
                intBG = self.vis_intensity_approx(0, alpha_bg, wave, dlambda)

                vis[i,:] = ((intSgrA + f1_fr*intF1 + s2_fr*intS2)/
                            (intSgrA_center + f1_fr*intF1_center + 
                            s2_fr*intS2_center + fluxRatioBG*intBG))
        visphi = np.angle(vis, deg=True)
        visphi = visphi + 360.*(visphi<-180.) - 360.*(visphi>180.)  
        
        if ret_flat:
            return visphi.flatten()[mask]
        else:
            return visphi
        
        
    def lnprob_unary(self, theta, visphif, visphierrf, 
                     mask, uv, wave, dlambda, sources, lower, upper):
        if np.any(theta < lower) or np.any(theta > upper):
            return -np.inf
        return self.lnlike_unary(theta, visphif, visphierrf, mask, uv, wave, dlambda, sources)
    
    
    def lnlike_unary(self, theta, visphi, visphi_error, mask, uv, wave, dlambda, sources):
        model_visphi = self.threesource(uv, wave, dlambda, sources, theta[0], theta[1], mask=mask)
        
        res_phi = (-np.minimum((model_visphi-visphi)**2.,
                                     (360-(model_visphi-visphi))**2.)/
                          visphi_error**2.)
        res_phi = np.sum(res_phi[~np.isnan(res_phi)])
        return 0.5*res_phi
        
        
        
    def fit_threesource(self, u, v, wave, dlambda, visphi, visphierr, header, sg_fr, s2_pos, plot=False, mcmc=False):
        phasemaps = self.fit_phasemaps
        uv = [u.flatten(),v.flatten()]
        visphif = visphi.flatten()
        visphierrf = visphierr.flatten()
        mask = ~np.isnan(visphif) * ~np.isnan(visphierrf)

        s2_fr = 1/sg_fr
        f1_pos = np.array([-18.78, 19.80])
        f1_fr = 10**(-(18.7-14.1)/2.5)*s2_fr
        
        if phasemaps:
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
            
            pmfile = resource_filename('gravipy', 'GRAVITY_SC_MAP_20200306_SM45.fits')
            phasemaps = fits.open(pmfile)
            pm_amp = phasemaps['SC_AMP'].data
            pm_pha = phasemaps['SC_PHASE'].data

            x = np.arange(201)
            y = np.arange(201)
            pm_amp_int = []
            pm_pha_int = []        
            for idx in range(4):
                amp = pm_amp[idx]
                amp /= np.max(amp)
                amp_mod = np.copy(amp)
                amp_mod[np.isnan(amp)] = 0
                pm_amp_int.append(interpolate.interp2d(x, y, amp_mod))

                pha = pm_pha[idx]
                pha_mod = np.copy(pha)
                pha_mod[np.isnan(pha)] = 0
                pm_pha_int.append(interpolate.interp2d(x, y, pha_mod))

            ra = s2_pos[0]
            dec = s2_pos[1]
            lambda0 = 2.2 
            cor_amp_s2 = np.ones((4, len(wave)))
            cor_pha_s2 = np.zeros((4, len(wave)))
            for tel in range(4):
                pos = np.array([ra + dra[tel], dec + ddec[tel]])
                pos_rot = np.dot(rotation(northangle[tel]), pos)
                for channel in range(len(wave)):
                    pos_scaled = pos_rot*lambda0/wave[channel] + 100
                    cor_amp_s2[tel, channel] = pm_amp_int[tel](pos_scaled[0], pos_scaled[1])
                    cor_pha_s2[tel, channel] = pm_pha_int[tel](pos_scaled[0], pos_scaled[1])

            
            ra = f1_pos[0]
            dec = f1_pos[1]
            cor_amp_f1 = np.ones((4, len(wave)))
            cor_pha_f1 = np.zeros((4, len(wave)))
            for tel in range(4):
                pos = np.array([ra + dra[tel], dec + ddec[tel]])
                pos_rot = np.dot(rotation(northangle[tel]), pos)
                for channel in range(len(wave)):
                    pos_scaled = pos_rot*lambda0/wave[channel] + 100
                    cor_amp_f1[tel, channel] = pm_amp_int[tel](pos_scaled[0], pos_scaled[1])
                    cor_pha_f1[tel, channel] = pm_pha_int[tel](pos_scaled[0], pos_scaled[1])
                    
            cor = [cor_amp_s2, cor_pha_s2, cor_amp_f1, cor_pha_f1]
            sources = [s2_pos, s2_fr, f1_pos, f1_fr, cor]
            
        else:
            cor_amp_s2 = np.ones((4, len(wave)))
            cor_pha_s2 = np.zeros((4, len(wave)))
            cor_amp_f1 = np.ones((4, len(wave)))
            cor_pha_f1 = np.zeros((4, len(wave)))
            fiber_coup = np.exp(-1*(2*np.pi*np.sqrt(np.sum(s2_pos**2))/280)**2)
            s2_fr = s2_fr * fiber_coup
            fiber_coup = np.exp(-1*(2*np.pi*np.sqrt(np.sum(f1_pos**2))/280)**2)
            f1_fr = f1_fr * fiber_coup

            sources = [s2_pos, s2_fr, f1_pos, f1_fr]
            
            
        if mcmc:
            nwalkers = 200
            nruns = 200
            size = 10
            
            popt1, pcov = optimize.curve_fit(lambda uv, x, y: self.pointsource(uv, wave, x, y, mask),
                                            uv, visphif[mask], sigma=visphierrf[mask],
                                            bounds=(-10,10))
            theta = np.array(popt1)
            theta_lower = np.array([-size, -size])
            theta_upper = np.array([size, size])
            theta_names = np.array([r"$RA_{PC}$", r"$DEC_{PC}$"])
            theta_names_raw = np.array(["PC RA", "PC DEC"])
            ndim = len(theta)
            width = 1e-1
            pos = np.ones((nwalkers,ndim))
            for par in range(ndim):
                pos[:,par] = theta[par] + width*np.random.randn(nwalkers)
                
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob_unary,
                                            args=(visphif[mask], visphierrf[mask], mask, 
                                                  uv, wave, dlambda, sources,
                                                  theta_lower, theta_upper))
            sampler.run_mcmc(pos, nruns, progress=False)
            samples = sampler.chain
            mostprop = sampler.flatchain[np.argmax(sampler.flatlnprobability)]

            fl_samples = samples[:, -nruns//2:, :].reshape((-1, ndim))
            if plot:
                fig, axes = plt.subplots(ndim, figsize=(8, ndim/1.5),
                                        sharex=True)
                for i in range(ndim):
                    ax = axes[i]
                    ax.plot(samples[:, :, i].T, "k", alpha=0.3)
                    ax.set_ylabel(theta_names[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number")
                plt.show()
                
                
                
                ranges = np.percentile(fl_samples, [3, 97], axis=0).T
                fig = corner.corner(fl_samples, quantiles=[0.16, 0.5, 0.84],
                                    truths=mostprop, labels=theta_names)
                plt.show()
            popt = np.percentile(fl_samples, [50], axis=0).T
            
            del sampler, samples
            gc.collect()
        
        else:
            popt1, pcov = optimize.curve_fit(lambda uv, x, y: self.pointsource(uv, wave, x, y, mask),
                                            uv, visphif[mask], sigma=visphierrf[mask],
                                            bounds=(-10,10))
            
            p0 = [popt1[0], popt1[1]]

            popt, pcov = optimize.curve_fit(lambda uv, x, y: self.threesource(uv, wave, dlambda, sources, x, y,
                                                                            mask=mask), 
                                            uv, visphif[mask], sigma=visphierrf[mask], 
                                            bounds=(-10,10), p0=p0)#, method="dogbox",**{"loss":'cauchy'})
            
        popt_res = self.threesource(uv, wave, dlambda, sources, *popt, mask=mask)
        
        chi = np.sum((visphif[mask] - popt_res)**2/visphierrf[mask]**2)
        ndof = len(popt_res)-2
        redchi = chi/ndof

        if plot:
            rad2as = 180 / np.pi * 3600
            wave_model = np.linspace(wave[0],wave[len(wave)-1],1000)
            dlambda_model = np.interp(wave_model, wave, dlambda)
            u_as = np.zeros((len(u),len(wave)))
            v_as = np.zeros((len(v),len(wave))) 
            u_as_model = np.zeros((len(u),len(wave_model)))
            v_as_model = np.zeros((len(v),len(wave_model)))
            for i in range(0,len(u)):
                u_as[i,:] = u[i]/(wave*1.e-6) / rad2as
                v_as[i,:] = v[i]/(wave*1.e-6) / rad2as
                u_as_model[i,:] = u[i]/(wave_model*1.e-6) / rad2as
                v_as_model[i,:] = v[i]/(wave_model*1.e-6) / rad2as
            magu_as_model = np.sqrt(u_as_model**2.+v_as_model**2.)
            magu_as = np.sqrt(u_as**2.+v_as**2.)   
            
            if phasemaps:
                ra = s2_pos[0]
                dec = s2_pos[1]
                lambda0 = 2.2 
                cor_amp_s2 = np.ones((4, len(wave_model)))
                cor_pha_s2 = np.zeros((4, len(wave_model)))
                for tel in range(4):
                    pos = np.array([ra + dra[tel], dec + ddec[tel]])
                    pos_rot = np.dot(rotation(northangle[tel]), pos)
                    for channel in range(len(wave_model)):
                        pos_scaled = pos_rot*lambda0/wave_model[channel] + 100
                        cor_amp_s2[tel, channel] = pm_amp_int[tel](pos_scaled[0], pos_scaled[1])
                        cor_pha_s2[tel, channel] = pm_pha_int[tel](pos_scaled[0], pos_scaled[1])
                ra = f1_pos[0]
                dec = f1_pos[1]
                cor_amp_f1 = np.ones((4, len(wave_model)))
                cor_pha_f1 = np.zeros((4, len(wave_model)))
                for tel in range(4):
                    pos = np.array([ra + dra[tel], dec + ddec[tel]])
                    pos_rot = np.dot(rotation(northangle[tel]), pos)
                    for channel in range(len(wave_model)):
                        pos_scaled = pos_rot*lambda0/wave_model[channel] + 100
                        cor_amp_f1[tel, channel] = pm_amp_int[tel](pos_scaled[0], pos_scaled[1])
                        cor_pha_f1[tel, channel] = pm_pha_int[tel](pos_scaled[0], pos_scaled[1])
                        
                cor = [cor_amp_s2, cor_pha_s2, cor_amp_f1, cor_pha_f1]
                sources = [s2_pos, s2_fr, f1_pos, f1_fr, cor]

            else:
                sources = [s2_pos, s2_fr, f1_pos, f1_fr]
            
            model_visphi = self.threesource(uv, wave_model, dlambda_model, sources,
                                    popt[0], popt[1], mask=1, ret_flat=False)
            
            for i in range(0,6):
                plt.errorbar(magu_as[i,:], visphi[i,:], visphierr[i,:], 
                            color=colors_baseline[i], label=baseline[i], 
                            ls='', lw=1, alpha=0.5, capsize=0)
                plt.scatter(magu_as[i,:], visphi[i,:],
                            color=colors_baseline[i], alpha=0.5)
                plt.plot(magu_as_model[i,:], model_visphi[i,:],
                        color='k', zorder=100)
            plt.legend()
            plt.ylabel('visibility phase [deg]')
            plt.xlabel('spatial frequency [1/arcsec]')
            plt.show()
        return popt[0], popt[1], redchi
        
        


    def fit_night_3src(self, plot=True, plotfits=False, phasemaps=False, only_sgr=False, ret_flux=True, cut_low=2, cut_up=2,
                       mcmc=False, nthreads=1):
        """
        Fit a 3 source model to all data from the night
        """
        self.fit_phasemaps = phasemaps
        ndit = self.ndit
        wave = self.wave
        dlambda = self.dlambda
        sg_files = self.sg_files
        s2_files = self.s2_files
        sg_u_raw = self.sg_u_raw
        sg_v_raw = self.sg_v_raw
        s2_u_raw = self.s2_u_raw
        s2_v_raw = self.s2_v_raw

        [[sg_t, sg_lst, sg_ang, sg_visphi_p1, sg_visphi_err_p1, sg_visphi_p2, sg_visphi_err_p2],
         [s2_t, s2_lst, s2_ang, s2_visphi_p1, s2_visphi_err_p1, s2_visphi_p2, s2_visphi_err_p2]] = self.alldata
        
        if not only_sgr:
            s2_ra_p1 = np.zeros((len(s2_files), ndit))*np.nan
            s2_de_p1 = np.zeros((len(s2_files), ndit))*np.nan
            s2_ra_p2 = np.zeros((len(s2_files), ndit))*np.nan
            s2_de_p2 = np.zeros((len(s2_files), ndit))*np.nan

            for fdx, file in enumerate(s2_files):
                for dit in range(ndit):
                    u = s2_u_raw[fdx, dit*6:(dit+1)*6]
                    v = s2_v_raw[fdx, dit*6:(dit+1)*6]
                    if np.sum(u==0) > 0:
                        continue

                    visphi = s2_visphi_p1[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    visphierr = s2_visphi_err_p1[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    wcut = np.copy(wave)[cut_low:-cut_up]
                    if np.sum(np.isnan(visphi)) > 10:
                        continue
                    s2_ra_p1[fdx, dit], s2_de_p1[fdx, dit] = self.fit_pointsource(u,v,wcut,visphi,visphierr,plot=plotfits)
                    
                    visphi = s2_visphi_p2[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    visphierr = s2_visphi_err_p2[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    wcut = np.copy(wave)[cut_low:-cut_up]
                    if np.sum(np.isnan(visphi)) > 10:
                        continue
                    s2_ra_p2[fdx, dit], s2_de_p2[fdx, dit] = self.fit_pointsource(u,v,wcut,visphi,visphierr,plot=plotfits)

        sg_ra_p1 = np.zeros((len(sg_files), ndit))*np.nan
        sg_de_p1 = np.zeros((len(sg_files), ndit))*np.nan
        sg_ra_p2 = np.zeros((len(sg_files), ndit))*np.nan
        sg_de_p2 = np.zeros((len(sg_files), ndit))*np.nan
        sg_chi_p1 = np.zeros((len(sg_files), ndit))*np.nan
        sg_chi_p2 = np.zeros((len(sg_files), ndit))*np.nan
        
        sg_flux = self.sg_flux
        s2_lpos = self.s2_pos
        
        if nthreads == 1:
            for fdx, file in enumerate(sg_files):
                sg_fr = sg_flux[fdx]
                s2_pos = s2_lpos[fdx]
                header = self.sg_header[fdx]
                if np.isnan(sg_fr):
                    if self.verbose:
                        print('SgrA* flux not available for %s' % sg_files[fdx])
                    sg_ra_p1[fdx] = np.nan
                    sg_de_p1[fdx] = np.nan
                    sg_ra_p2[fdx] = np.nan
                    sg_de_p2[fdx] = np.nan
                    continue
                
                for dit in range(ndit):
                    u = sg_u_raw[fdx, dit*6:(dit+1)*6]
                    v = sg_v_raw[fdx, dit*6:(dit+1)*6]
                    if np.sum(u==0) > 0:
                        continue

                    visphi = sg_visphi_p1[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    visphierr = sg_visphi_err_p1[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    wcut = np.copy(wave)[cut_low:-cut_up]
                    dwcut = np.copy(dlambda)[cut_low:-cut_up]
                    if np.sum(np.isnan(visphi)) > 10:
                        continue
                    sg_ra_p1[fdx, dit], sg_de_p1[fdx, dit], sg_chi_p1[fdx, dit] = self.fit_threesource(u,v,wcut,dwcut,
                                                                            visphi,visphierr,header, sg_fr, s2_pos, 
                                                                            plot=plotfits, mcmc=mcmc)
                    
                    visphi = sg_visphi_p2[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    visphierr = sg_visphi_err_p2[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    wcut = np.copy(wave)[cut_low:-cut_up]
                    dwcut = np.copy(dlambda)[cut_low:-cut_up]
                    if np.sum(np.isnan(visphi)) > 10:
                        continue
                    sg_ra_p2[fdx, dit], sg_de_p2[fdx, dit], sg_chi_p2[fdx, dit]  = self.fit_threesource(u,v,wcut,dwcut,
                                                                            visphi,visphierr,header, sg_fr, s2_pos, 
                                                                            plot=plotfits, mcmc=mcmc)
                    
        else:
            def _fit_file(fdx):
                print(fdx)
                file = sg_files[fdx]
                sg_fr = sg_flux[fdx]
                s2_pos = s2_lpos[fdx]
                header = self.sg_header[fdx]
                
                _ra_p1 = np.zeros(ndit)*np.nan
                _de_p1 = np.zeros(ndit)*np.nan
                _ra_p2 = np.zeros(ndit)*np.nan
                _de_p2 = np.zeros(ndit)*np.nan
                _chi_p1 = np.zeros(ndit)*np.nan
                _chi_p2 = np.zeros(ndit)*np.nan
                
                if np.isnan(sg_fr):
                    if self.verbose:
                        print('SgrA* flux not available for %s' % sg_files[fdx])
                    _ra_p1 = np.nan
                    _de_p1 = np.nan
                    _ra_p2 = np.nan
                    _de_p2 = np.nan
                    _res = np.array([_ra_p1, _de_p1, _ra_p2, _de_p2])
                    return _res
                
                for dit in range(ndit):
                    u = sg_u_raw[fdx, dit*6:(dit+1)*6]
                    v = sg_v_raw[fdx, dit*6:(dit+1)*6]
                    if np.sum(u==0) > 0:
                        continue

                    visphi = sg_visphi_p1[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    visphierr = sg_visphi_err_p1[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    wcut = np.copy(wave)[cut_low:-cut_up]
                    dwcut = np.copy(dlambda)[cut_low:-cut_up]
                    if np.sum(np.isnan(visphi)) > 10:
                        continue
                    _ra_p1[dit], _de_p1[dit], _chi_p1[dit] = self.fit_threesource(u,v,wcut,dwcut,
                                                                    visphi,visphierr,header, sg_fr, s2_pos,
                                                                    plot=plotfits, mcmc=mcmc)
                    
                    visphi = sg_visphi_p2[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    visphierr = sg_visphi_err_p2[fdx, dit*6:(dit+1)*6][:,cut_low:-cut_up]
                    wcut = np.copy(wave)[cut_low:-cut_up]
                    dwcut = np.copy(dlambda)[cut_low:-cut_up]
                    if np.sum(np.isnan(visphi)) > 10:
                        continue
                    _ra_p2[dit], _de_p2[dit], _chi_p2[dit] = self.fit_threesource(u,v,wcut,dwcut,
                                                                    visphi,visphierr,header, sg_fr, s2_pos,
                                                                    plot=plotfits, mcmc=mcmc)
                _res = np.array([_ra_p1, _de_p1, _ra_p2, _de_p2, _chi_p1, _chi_p2])
                return _res


            #pool = multiprocessing.Pool(nthreads)
            #mcoreres = np.array(pool.map(_fit_file, np.arange(len(sg_files))))
            
            #mcoreres = []
            #for f in range(len(sg_files)):
                #print(f)
                #mcoreres.append(_fit_file(f))
            #mcoreres = np.asarray(mcoreres)
            mcoreres = np.array(Parallel(n_jobs=nthreads, 
                                         verbose=51,
                                         #backend="threading"
                                         )(delayed(_fit_file)(fdx) for fdx in range(len(sg_files))))
        
            sg_ra_p1 = mcoreres[:,0,:]
            sg_de_p1 = mcoreres[:,1,:]
            sg_ra_p2 = mcoreres[:,2,:]
            sg_de_p2 = mcoreres[:,3,:]
            sg_chi_p1 = mcoreres[:,4,:]
            sg_chi_p2 = mcoreres[:,5,:]
                
                  
        if plot:
            if ndit == 1:
                umark = 'o'
            else:
                umark = '.'
            plt.figure(figsize=(7,6))
            gs = gridspec.GridSpec(3,1, hspace=0.05)
            axis = plt.subplot(gs[0,0])
            plt.title('Position fit')
            plt.plot(sg_t.flatten()[::ndit], sg_flux, 
                     color=color1, ls='-',lw=0.5, marker='o')
            axis.set_xticklabels([])
            plt.ylabel('Flux [S2]')
            plt.ylim(0,1.5)
            axis = plt.subplot(gs[1,0])
            if not only_sgr:
                plt.plot(s2_t.flatten(), (s2_ra_p1.flatten()+s2_ra_p2.flatten())/2, 
                        color='k', ls='', marker=umark, label='S2')
            plt.plot(sg_t.flatten(), (sg_ra_p1.flatten()+sg_ra_p2.flatten())/2, 
                     color=color1, ls='', marker=umark, label='SgrA* (3src fit)')
            plt.axhline(0, color='grey', lw=0.5, zorder=0)
            plt.legend()
            axis.set_xticklabels([])
            plt.ylabel('RA [mas]')
            axis = plt.subplot(gs[2,0])
            if not only_sgr:
                plt.plot(s2_t.flatten(), (s2_de_p1.flatten()+s2_de_p2.flatten())/2, 
                        color='k', ls='', marker=umark)
            plt.plot(sg_t.flatten(), (sg_de_p1.flatten()+sg_de_p2.flatten())/2, 
                     color=color1, ls='', marker=umark)
            plt.axhline(0, color='grey', lw=0.5, zorder=0)
            plt.xlabel('Time [min]')
            plt.ylabel('Dec [mas]')
            plt.show()

        if not only_sgr:
            fitres = [[sg_t, sg_ra_p1, sg_de_p1, sg_ra_p2, sg_de_p2],
                      [s2_t, s2_ra_p1, s2_de_p1, s2_ra_p2, s2_de_p2]]
        else:
            fitres = [sg_t, sg_ra_p1, sg_de_p1, sg_ra_p2, sg_de_p2, sg_chi_p1, sg_chi_p2]
        if ret_flux:
            return sg_flux, fitres
        else:
            return fitres



    def vis_onestar(self, uv, wave, dlambda, sources, alpha=3):
        mas2rad = 1e-3 / 3600 / 180 * np.pi
        u = uv[0]
        v = uv[1]
        pos, fl = sources
        vis = np.zeros((6,len(wave))) + 0j
        for i in range(0,6):
            s = ((pos[0])*u[i] + (pos[1])*v[i]) * mas2rad * 1e6

            int_s = self.vis_intensity_approx(s, alpha, wave, dlambda)
            vis[i,:] = fl*int_s
        return vis

    
    def night_remove_sources(self):
        
        fitres = self.fit_night_3src(plot=False, only_sgr=True, ret_flux=False)
        [_, sg_ra_p1, sg_de_p1, sg_ra_p2, sg_de_p2, _, _] = fitres

        ndit = self.ndit
        wave = self.wave
        dlambda = self.dlambda
        sg_files = self.sg_files
        s2_files = self.s2_files
        sg_u_raw = self.sg_u_raw
        sg_v_raw = self.sg_v_raw
        s2_u_raw = self.s2_u_raw
        s2_v_raw = self.s2_v_raw
        
        [[sg_t, sg_lst, sg_ang, sg_visphi_p1, sg_visphi_err_p1, sg_visphi_p2, sg_visphi_err_p2],
         [s2_t, s2_lst, s2_ang, s2_visphi_p1, s2_visphi_err_p1, s2_visphi_p2, s2_visphi_err_p2]] = self.alldata
        
        sg_flux = self.sg_flux
        s2_lpos = self.s2_pos
        
        sg_visphi_p1_corr = np.copy(sg_visphi_p1)
        sg_visphi_p2_corr = np.copy(sg_visphi_p2)

        for fdx, file in enumerate(sg_files):
            sg_fr = sg_flux[fdx]
            s2_pos = s2_lpos[fdx]
            header = self.sg_header[fdx]
            if np.isnan(sg_fr):
                if self.verbose:
                    print('SgrA* flux not available for %s' % sg_files[fdx])
                continue
            
            for dit in range(ndit):
                u = sg_u_raw[fdx, dit*6:(dit+1)*6]
                v = sg_v_raw[fdx, dit*6:(dit+1)*6]
                uv = [u.flatten(),v.flatten()]
                visphi_p1 = sg_visphi_p1[fdx, dit*6:(dit+1)*6]
                visphi_p2 = sg_visphi_p2[fdx, dit*6:(dit+1)*6]
                
                s2_fr = 1/sg_fr
                f1_pos = np.array([-18.78, 19.80])
                f1_fr = 10**(-(18.7-14.1)/2.5)*s2_fr
                fiber_coup = np.exp(-1*(2*np.pi*np.sqrt(np.sum(s2_pos**2))/280)**2)
                s2_fr = s2_fr * fiber_coup
                fiber_coup = np.exp(-1*(2*np.pi*np.sqrt(np.sum(f1_pos**2))/280)**2)
                f1_fr = f1_fr * fiber_coup
                
                sources = [s2_pos, s2_fr, f1_pos, f1_fr]
                
                # p1
                visphi_3src = self.threesource(uv, wave, dlambda, sources,
                                               sg_ra_p1[fdx,dit], sg_de_p1[fdx,dit], ret_flat=False)
                visphi_1src = self.pointsource(uv, wave, sg_ra_p1[fdx,dit], sg_de_p1[fdx,dit], flatten=False)
                visphi_res = visphi_3src - visphi_1src
                visphi_p1_cor = visphi_p1 - visphi_res
                visphi_p1_cor = (visphi_p1_cor+180)%360-180

                # p2
                visphi_3src = self.threesource(uv, wave, dlambda, sources,
                                               sg_ra_p2[fdx,dit], sg_de_p2[fdx,dit], ret_flat=False)
                visphi_1src = self.pointsource(uv, wave, sg_ra_p2[fdx,dit], sg_de_p2[fdx,dit], flatten=False)
                visphi_res = visphi_3src - visphi_1src
                visphi_p2_cor = visphi_p2 - visphi_res
                visphi_p2_cor = (visphi_p2_cor+180)%360-180
                
                sg_visphi_p1_corr[fdx, dit*6:(dit+1)*6] = visphi_p1_cor
                sg_visphi_p2_corr[fdx, dit*6:(dit+1)*6] = visphi_p2_cor
        return sg_visphi_p1_corr, sg_visphi_p2_corr
                
                


