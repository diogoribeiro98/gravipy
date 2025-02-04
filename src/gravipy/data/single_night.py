import os
import numpy as np

#Fits utils
from astropy.io import fits

#Logging tools
import logging
from ..logger.log import log_level_mapping

#Time tools
from ..tools.time_and_dates import convert_date

#Plot utils
import matplotlib.pyplot as plt
from matplotlib import gridspec

fw_black = 'k'
fw_dblue = 'darkblue'
fw_lblue = '#006BA4'
fw_gray = '#595959'
fw_orange = '#FF800E' 
fw_red = 'darkred'

class GravData():
    """GRAVITY data loader
    """

    def __init__(self, 
                 data, 
                 loglevel='INFO',
                 plot=False,
                 flag=True,
                 ignore_telescopes = []
                 ):

        #Create a logger and set log level according to user
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(log_level_mapping.get(loglevel, logging.INFO))
        
        # ---------------------------
        # Pre-define class quantities
        # ---------------------------
        
        self.name = data
        self.filename   =  os.path.basename(data)
        
        self.header     = None
        self.datacatg   = None
        
        self.date_obs   = None
        self.date       = None
        self.mjd        = None

        self.polmode    = None 
        self.resolution = None
        self.dit        = None
        self.ndit       = None

        self.tel        = None

        self.baseline_labels = None
        self.closure_labels  = None
        self.baseline_telescopes = None

        self.wlSC    = None
        self.wlSC_nchannels = None

        self.wlFT    = None
        self.wFTC_nchannels = None

        self.tel_pos = None

        self.u = None
        self.v =  None
        self.spatial_frequency_as = None
        self.spatial_frequency_as_T3 = None
        self.bispec_ind = None
        
        self.visampSC_P1    = None
        self.visamperrSC_P1 = None
        self.visphiSC_P1    = None
        self.visphierrSC_P1 = None
        self.vis2SC_P1      = None
        self.vis2errSC_P1   = None
        self.t3SC_P1        = None
        self.t3errSC_P1     = None
        self.t3ampSC_P1     = None
        self.t3amperrSC_P1  = None

        self.visampSC_P2    = None
        self.visamperrSC_P2 = None
        self.visphiSC_P2    = None
        self.visphierrSC_P2 = None
        self.vis2SC_P2      = None
        self.vis2errSC_P2   = None
        self.t3SC_P2        = None
        self.t3errSC_P2     = None
        self.t3ampSC_P2     = None
        self.t3amperrSC_P2  = None

        self.visampflagSC_P1 = None
        self.visampflagSC_P2 = None
        self.vis2flagSC_P1   = None
        self.vis2flagSC_P2   = None
        self.t3flagSC_P1     = None
        self.t3flagSC_P2     = None
        self.t3ampflagSC_P1  = None
        self.t3ampflagSC_P2  = None
        self.visphiflagSC_P1 = None
        self.visphiflagSC_P2 = None

        self.colors_baseline = np.array([fw_black , fw_dblue , fw_gray, fw_orange, fw_red, fw_lblue])
        self.colors_closure  = np.array([fw_lblue , fw_red   , fw_black, fw_orange])

        #List of allowed file types
        allowed_file_types = [
            'VIS_DUAL_SCI_RAW', 
            'VIS_SINGLE_SCI_RAW',
            'VIS_SINGLE_CAL_RAW', 
            'VIS_DUAL_CAL_RAW',
            'VIS_SINGLE_CALIBRATED', 
            'VIS_DUAL_CALIBRATED',
            'SINGLE_SCI_VIS', 
            'SINGLE_SCI_VIS_CALIBRATED',
            'DUAL_SCI_VIS', 
            'DUAL_SCI_VIS_CALIBRATED',
            'SINGLE_CAL_VIS', 
            'DUAL_CAL_VIS', 
            'ASTROREDUCED',
            'DUAL_SCI_P2VMRED',
            ]

        # ---------------------------
        # Load Header data
        # ---------------------------

        #Get header information
        with fits.open(self.name) as hdul:
            self.header = hdul[0].header

        #Check datafile instrument
        if 'GRAV' not in self.header['INSTRUME']:
            raise ValueError('File seems to be not from GRAVITY')
        
        # Note: The keyword ESO PRO CATG stands for ESO PROCESSED CATEGORY!
        if 'ESO PRO CATG' in self.header:
            self.datacatg = self.header['ESO PRO CATG']
            if self.datacatg not in allowed_file_types:
                raise ValueError('filetype is %s, which is not supported' % self.datacatg)

        #Load header data
        self.date_obs   = self.header['DATE-OBS']
        self.date       = convert_date(self.date_obs)
        self.mjd        = convert_date(self.date_obs, mjd=True)

        self.polmode    = self.header['ESO INS POLA MODE']
        self.resolution = self.header['ESO INS SPEC RES']
        self.dit        = self.header['ESO DET2 SEQ1 DIT']
        self.ndit       = self.header['ESO DET2 NDIT']

        if self.header['TELESCOP'] in ['ESO-VLTI-U1234', 'U1234']:
            self.tel = 'UT'
        elif self.header['TELESCOP'] in ['ESO-VLTI-A1234', 'A1234']:
            self.tel = 'AT'
        else:
            raise ValueError('Telescope not AT or UT, seomtehign wrong with input data')
        
        # --------------------------------------
        # Load baseline information from header
        # --------------------------------------

        baseline_labels = []
        closure_labels  = []
        baseline_telescopes = []

        tel_name  = fits.open(self.name)['OI_ARRAY'].data['TEL_NAME']
        sta_index = fits.open(self.name)['OI_ARRAY'].data['STA_INDEX']
        tel_pos  = fits.open(self.name)['OI_ARRAY'].data['STAXYZ'][:,0:2]

        
        if self.polmode == 'SPLIT':
                vis_index = fits.open(self.name)['OI_VIS', 11].data['STA_INDEX']
                t3_index  = fits.open(self.name)['OI_T3' , 11].data['STA_INDEX']
        else:
            vis_index = fits.open(self.name)['OI_VIS', 10].data['STA_INDEX']
            t3_index  = fits.open(self.name)['OI_T3' , 10].data['STA_INDEX']
        
        for bl in range(6):
            t1 = np.where(sta_index == vis_index[bl, 0])[0][0]
            t2 = np.where(sta_index == vis_index[bl, 1])[0][0]
            baseline_labels.append(tel_name[t1] + '-' + tel_name[t2][2])
            baseline_telescopes.append( (tel_name[t1], tel_name[t2]))
        
        for cl in range(4):
            t1 = np.where(sta_index == t3_index[cl, 0])[0][0]
            t2 = np.where(sta_index == t3_index[cl, 1])[0][0]
            t3 = np.where(sta_index == t3_index[cl, 2])[0][0]
            closure_labels.append(tel_name[t1] + '-' + tel_name[t2][2] + '-' + tel_name[t3][2])
        
        self.baseline_labels = np.array(baseline_labels)
        self.closure_labels  = np.array(closure_labels)
        self.baseline_telescopes = np.array(baseline_telescopes)
        self.tel_pos = np.array(tel_pos)

        # ----------------------------
        # Load wavelength data
        # ----------------------------

        if self.polmode == 'COMBINED':
            
            self.wlSC = fits.open(self.name)['OI_WAVELENGTH', 10].data['EFF_WAVE']*1e6
            self.wlSC_nchannels = len(self.wlSC)

            if not (self.datacatg == 'ASTROREDUCED'):
                self.wlFT = fits.open(self.name)['OI_WAVELENGTH', 20].data['EFF_WAVE']*1e6
                self.wlFT_nchannels = len(self.wlFT)

            effband = fits.open(self.name)['OI_WAVELENGTH', 10].data['EFF_BAND']
            self.dlambda = effband/2*1e6

        elif self.polmode == 'SPLIT':

            wlSC_P1 = fits.open(self.name)['OI_WAVELENGTH', 11].data['EFF_WAVE']*1e6
            #wlSC_P2 = fits.open(self.name)['OI_WAVELENGTH', 12].data['EFF_WAVE']*1e6
            
            self.wlSC = wlSC_P1
            self.wlSC_nchannels = len(self.wlSC)
            
            if not (self.datacatg == 'ASTROREDUCED'):
                wlFT_P1 = fits.open(self.name)['OI_WAVELENGTH', 21].data['EFF_WAVE']*1e6
                #wlFT_P2 = fits.open(self.name)['OI_WAVELENGTH', 22].data['EFF_WAVE']*1e6
                
                self.wlFT = wlFT_P1
                self.wlFT_nchannels = len(self.wlFT)

            effband = fits.open(self.name)['OI_WAVELENGTH', 11].data['EFF_BAND']
            self.dlambda = effband/2*1e6

        else:
            raise ValueError(f'Polarization mode {self.polmode} not supported. Must be SPLIT or COMBINED')

        # ----------------------------
        # Load interferometric data
        # ----------------------------
        
        if self.polmode == 'COMBINED':
            raise ValueError(f'Polarization mode {self.polmode} currently not implemented')

        elif self.polmode == 'SPLIT':
        
            #Open file
            fitsdata = fits.open(self.name)

            #Get UV coordinates
            self.u = fitsdata['OI_VIS', 11].data.field('UCOORD')
            self.v = fitsdata['OI_VIS', 11].data.field('VCOORD')

            #Convert UV coordinates to inverse arcseconds
            u_as = np.zeros((len(self.u), self.wlSC_nchannels))
            v_as = np.zeros((len(self.v), self.wlSC_nchannels))
                
            for i in range(0, len(self.u)):
                u_as[i, :] = (self.u[i] / (self.wlSC * 1.e-6) * np.pi / 180. / 3600.)  # 1/as
                v_as[i, :] = (self.v[i] / (self.wlSC * 1.e-6) * np.pi / 180. / 3600.)  # 1/as

            self.spatial_frequency_as = np.sqrt(u_as**2.+v_as**2.)
            
            uv_magnitude = np.sqrt(self.u**2.+self.v**2.)

            #Note: For closure phases, it is custumary to use as reference the longest baseline
            max_spf = np.zeros(4)

            max_spf[0] = np.max(np.array([uv_magnitude[0], uv_magnitude[3], uv_magnitude[1]]))
            max_spf[1] = np.max(np.array([uv_magnitude[0], uv_magnitude[4], uv_magnitude[2]]))
            max_spf[2] = np.max(np.array([uv_magnitude[1], uv_magnitude[5], uv_magnitude[2]]))
            max_spf[3] = np.max(np.array([uv_magnitude[3], uv_magnitude[5], uv_magnitude[4]]))

            self.spatial_frequency_as_T3 = np.zeros((len(max_spf),  self.wlSC_nchannels))
            
            for idx in range(len(max_spf)):
                self.spatial_frequency_as_T3[idx] = (max_spf[idx]/(self.wlSC*1.e-6) * np.pi / 180. / 3600.)  # 1/as

            #Closure phase bispectrum indexes
            self.bispec_ind = np.array([[0, 3, 1],
                                        [0, 4, 2],
                                        [1, 5, 2],
                                        [3, 5, 4]])

            #Load data

            # P1
            self.visampSC_P1    = fitsdata['OI_VIS',  11].data.field('VISAMP')
            self.visamperrSC_P1 = fitsdata['OI_VIS',  11].data.field('VISAMPERR')
            self.visphiSC_P1    = fitsdata['OI_VIS',  11].data.field('VISPHI')
            self.visphierrSC_P1 = fitsdata['OI_VIS',  11].data.field('VISPHIERR')
            self.vis2SC_P1      = fitsdata['OI_VIS2', 11].data.field('VIS2DATA')
            self.vis2errSC_P1   = fitsdata['OI_VIS2', 11].data.field('VIS2ERR')
            self.t3SC_P1        = fitsdata['OI_T3',   11].data.field('T3PHI')
            self.t3errSC_P1     = fitsdata['OI_T3',   11].data.field('T3PHIERR')
            self.t3ampSC_P1     = fitsdata['OI_T3',   11].data.field('T3AMP')
            self.t3amperrSC_P1  = fitsdata['OI_T3',   11].data.field('T3AMPERR')
            
            # P2
            self.visampSC_P2    = fitsdata['OI_VIS',  12].data.field('VISAMP')
            self.visamperrSC_P2 = fitsdata['OI_VIS',  12].data.field('VISAMPERR')
            self.visphiSC_P2    = fitsdata['OI_VIS',  12].data.field('VISPHI')
            self.visphierrSC_P2 = fitsdata['OI_VIS',  12].data.field('VISPHIERR')
            self.vis2SC_P2      = fitsdata['OI_VIS2', 12].data.field('VIS2DATA')
            self.vis2errSC_P2   = fitsdata['OI_VIS2', 12].data.field('VIS2ERR')
            self.t3SC_P2        = fitsdata['OI_T3',   12].data.field('T3PHI')
            self.t3errSC_P2     = fitsdata['OI_T3',   12].data.field('T3PHIERR')
            self.t3ampSC_P2     = fitsdata['OI_T3',   12].data.field('T3AMP')
            self.t3amperrSC_P2  = fitsdata['OI_T3',   12].data.field('T3AMPERR')

            # Flags
            self.visampflagSC_P1 = fitsdata['OI_VIS',  11].data.field('FLAG')
            self.visampflagSC_P2 = fitsdata['OI_VIS',  12].data.field('FLAG')
            self.vis2flagSC_P1   = fitsdata['OI_VIS2', 11].data.field('FLAG')
            self.vis2flagSC_P2   = fitsdata['OI_VIS2', 12].data.field('FLAG')
            self.t3flagSC_P1     = fitsdata['OI_T3',   11].data.field('FLAG')
            self.t3flagSC_P2     = fitsdata['OI_T3',   12].data.field('FLAG')
            self.t3ampflagSC_P1  = fitsdata['OI_T3',   11].data.field('FLAG')
            self.t3ampflagSC_P2  = fitsdata['OI_T3',   12].data.field('FLAG')
            self.visphiflagSC_P1 = fitsdata['OI_VIS',  11].data.field('FLAG')
            self.visphiflagSC_P2 = fitsdata['OI_VIS',  12].data.field('FLAG')

            #Ignore selected telescopes
            for t in ignore_telescopes:
                
                for cdx, cl in enumerate(self.closure_labels):
                    if str(t) in cl:
                        self.t3flagSC_P1[cdx]    = True
                        self.t3flagSC_P2[cdx]    = True
                        self.t3ampflagSC_P1[cdx] = True
                        self.t3ampflagSC_P2[cdx] = True
                
                for vdx, vi in enumerate(self.baseline_labels):
                    if str(t) in vi:
                        self.visampflagSC_P1[vdx] = True
                        self.visampflagSC_P2[vdx] = True
                        self.vis2flagSC_P1[vdx]   = True
                        self.vis2flagSC_P2[vdx]   = True
                        self.visphiflagSC_P1[vdx] = True
                        self.visphiflagSC_P2[vdx] = True

            #If data is flaged, put it to np.nan
            if flag:
                self.visampSC_P1[self.visampflagSC_P1]      = np.nan
                self.visamperrSC_P1[self.visampflagSC_P1]   = np.nan
                self.visampSC_P2[self.visampflagSC_P2]      = np.nan
                self.visamperrSC_P2[self.visampflagSC_P2]   = np.nan

                self.vis2SC_P1[self.vis2flagSC_P1]          = np.nan
                self.vis2errSC_P1[self.vis2flagSC_P1]       = np.nan
                self.vis2SC_P2[self.vis2flagSC_P2]          = np.nan
                self.vis2errSC_P2[self.vis2flagSC_P2]       = np.nan

                self.t3SC_P1[self.t3flagSC_P1]              = np.nan
                self.t3errSC_P1[self.t3flagSC_P1]           = np.nan
                self.t3SC_P2[self.t3flagSC_P2]              = np.nan
                self.t3errSC_P2[self.t3flagSC_P2]           = np.nan

                self.t3ampSC_P1[self.t3ampflagSC_P1]        = np.nan
                self.t3amperrSC_P1[self.t3ampflagSC_P1]     = np.nan
                self.t3ampSC_P2[self.t3ampflagSC_P2]        = np.nan
                self.t3amperrSC_P2[self.t3ampflagSC_P2]     = np.nan

                self.visphiSC_P1[self.visphiflagSC_P1]      = np.nan
                self.visphierrSC_P1[self.visphiflagSC_P1]   = np.nan
                self.visphiSC_P2[self.visphiflagSC_P2]      = np.nan
                self.visphierrSC_P2[self.visphiflagSC_P2]   = np.nan


            if plot:
                self.plot_interferometric_data()


        #Log information    
        self.logger.debug(f'Category: {self.datacatg}')
        self.logger.debug(f'Telescope: {self.tel}')
        self.logger.debug(f'Polarization: {self.polmode}')
        self.logger.debug(f'Resolution: {self.resolution}')
        self.logger.debug(f'DIT: {self.dit}')
        self.logger.debug(f'NDIT: {self.ndit}')

    
    def plot_interferometric_data(self,plotTAmp=False):
        #Define helper plot configurations
        plot_config = {
            'alpha':    0.8,
            'ms':       3.0,
            'lw':       0.8,
            'capsize':  1.0,
            'ls':       ''    
        }

        if plotTAmp:
            ncols = 5
        else:
            ncols = 4

        fig, axes = plt.subplots(ncols=ncols, figsize=(ncols*4.2,3))

        #Visibility Amplitude plot
        ax = axes[0]
        ax.set_title('Visibility Amplitude')
        ax.set_xlim(70, 320)
        ax.set_ylim(-0.0, 1.1)
        ax.set_ylabel('Visibility Amplitude')
        ax.set_xlabel('spatial frequency (1/arcsec)')
        ax.axhline(1, ls='--', lw=0.5)

        for idx in range(len(self.spatial_frequency_as)):
                        
            x   = self.spatial_frequency_as[idx] 
            y   = self.visampSC_P1[idx] 
            yerr= self.visamperrSC_P1[idx]

            ax.errorbar(x, y, yerr, **plot_config, marker='o', color=self.colors_baseline[idx % 6])
        
            y   = self.visampSC_P2[idx]   
            yerr= self.visamperrSC_P2[idx]

            ax.errorbar(x, y, yerr, **plot_config, marker='D', color=self.colors_baseline[idx % 6])

        #Visibility Phase plot
        ax = axes[1]
        ax.set_title('Visibility Phase')
        ax.set_xlim(70, 320)
        ax.set_ylim(-250, 250)
        ax.set_ylabel('Visibility Phase')
        ax.set_xlabel('spatial frequency (1/arcsec)')

        for idx in range(len(self.spatial_frequency_as)):
                        
            x   = self.spatial_frequency_as[idx] 
            y   = self.visphiSC_P1[idx] 
            yerr= self.visphierrSC_P1[idx]

            ax.errorbar(x, y, yerr, **plot_config, marker='o', color=self.colors_baseline[idx % 6])
        
            y   = self.visphiSC_P2[idx] 
            yerr= self.visphierrSC_P2[idx]

            ax.errorbar(x, y, yerr, **plot_config, marker='D', color=self.colors_baseline[idx % 6])

        #Visibility Squared plot
        ax = axes[2]
        ax.set_title('Visibility Squared')
        ax.set_xlim(70, 320)
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Visibility Squared')
        ax.set_xlabel('spatial frequency (1/arcsec)')
        ax.axhline(1, ls='--', lw=0.5)

        for idx in range(len(self.spatial_frequency_as)):
                        
            x   = self.spatial_frequency_as[idx] 
            y   = self.vis2SC_P1[idx] 
            yerr= self.vis2errSC_P1[idx]

            ax.errorbar(x, y, yerr, **plot_config, marker='o', color=self.colors_baseline[idx % 6])
        
            y   = self.vis2SC_P2[idx] 
            yerr= self.vis2errSC_P2[idx]

            ax.errorbar(x, y, yerr, **plot_config, marker='D', color=self.colors_baseline[idx % 6])
        
        #Closure phases
        ax = axes[3]
        ax.set_title('Closure Phases')
        ax.set_xlim(150, 320)
        ax.set_ylim(-150, 150)
        ax.set_ylabel('Closure Phases')
        ax.set_xlabel('spatial frequency (1/arcsec)')

        for idx in range(len(self.spatial_frequency_as_T3)):
                        
            x   = self.spatial_frequency_as_T3[idx] 
            y   = self.t3SC_P1[idx] 
            yerr= self.t3errSC_P1[idx]

            ax.errorbar(x, y, yerr, **plot_config, marker='o', color=self.colors_baseline[idx % 6])
        
            y   = self.t3SC_P2[idx] 
            yerr= self.t3errSC_P2[idx]

            ax.errorbar(x, y, yerr, **plot_config, marker='D', color=self.colors_baseline[idx % 6])
        
        if plotTAmp:
            #Closure Amplitudes
            ax = axes[4]
            ax.set_title('Closure Amplitude')
            ax.set_xlim(150, 320)
            ax.set_ylim(-0.1, 1.1)
            ax.set_ylabel('Closure Amplitude')
            ax.set_xlabel('spatial frequency (1/arcsec)')
            ax.axhline(1, ls='--', lw=0.5)

            for idx in range(len(self.spatial_frequency_as_T3)):
                            
                x   = self.spatial_frequency_as_T3[idx] 
                y   = self.t3ampSC_P1[idx] 
                yerr= self.t3amperrSC_P1[idx]

                ax.errorbar(x, y, yerr, **plot_config, marker='o', color=self.colors_baseline[idx % 6])
            
                y   = self.t3ampSC_P2[idx] 
                yerr= self.t3amperrSC_P2[idx]

                ax.errorbar(x, y, yerr, **plot_config, marker='D', color=self.colors_baseline[idx % 6])
            
        return fig, axes