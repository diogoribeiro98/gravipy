import mygravipy as gp
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import logging
from mygravipy.gravmfit import _calc_vis_mstars
from astropy.io import fits

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal

class LoggingHandler(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        log_message = self.format(record)
        self.text_edit.append(log_message)


class LoadFiles(QThread):
    finished = pyqtSignal()
    update_progress = pyqtSignal(int)

    def __init__(self, file_names):
        super().__init__()
        self.file_names = file_names
        self.files = []
        self.offs = []

    def run(self):
        for fdx, file in enumerate(self.file_names):
            # check if file is a fits file
            if file[-5:] != '.fits':
                logging.error(f'File {file} is not a fits file')
                continue
            h = fits.open(file)[0].header
            try:
                self.offs.append((h['ESO INS SOBJ OFFX'], h['ESO INS SOBJ OFFY']))
            except KeyError:
                logging.error(f'File {file} does not have OFFX/OFFY in header')
                continue
            self.files.append(file)
            self.update_progress.emit(fdx)
        self.finished.emit()


class LoadData(QThread):
    finished = pyqtSignal()
    update_progress = pyqtSignal(int)

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.data = None

    def run(self):
        data = gp.GravMFit(self.filename)
        data.get_int_data()
        self.data = data
        #     self.update_progress.emit(fdx)
        self.finished.emit()


class LoadDataList(QThread):
    finished = pyqtSignal()
    update_progress = pyqtSignal(int)

    def __init__(self, filenames):
        super().__init__()
        self.filenames = filenames
        self.data = []

    def run(self):
        for fdx, filename in enumerate(self.filenames):
            data = gp.GravMFit(filename, loglevel='WARNING')
            self.data.append(data)
            self.update_progress.emit(fdx)
        self.finished.emit()


class FitWorker(QThread):
    finished = pyqtSignal()
    update_progress = pyqtSignal(int)

    def __init__(self, data, input_dict, checkbox_dict, minimizer, nsources):
        super().__init__()
        self.data = data
        self.input_dict = input_dict
        self.checkbox_dict = checkbox_dict
        self.minimizer = minimizer
        self.nsources = nsources
        self.res = []

    def run(self):
        ra_list = []
        de_list = []
        fr_list = []
        fit_pos = []
        fit_fr = []
        for idx in range(self.nsources-1):
            ra_list.append(float(self.input_dict[f"RA {idx+1}"]))
            de_list.append(float(self.input_dict[f"Dec {idx+1}"]))
            fit_pos.append(self.checkbox_dict[f"pos {idx+1}"])
            if idx > 0:
                fr_list.append(float(self.input_dict[f"fr {idx+1}"]))
                fit_fr.append(self.checkbox_dict[f"fr {idx+1}"])

        initial = [float(self.input_dict["alphaBH"]),
                   float(self.input_dict["frBG"]),
                   float(self.input_dict["pcRA"]),
                   float(self.input_dict["pcDec"]),
                   float(self.input_dict["frBH"]),
                   1]
        
        for fdx, data in enumerate(self.data):
            if self.minimizer == 'Least Sqr':
                res = data.fit_stars(ra_list,
                                        de_list,
                                        fr_list,
                                        fit_pos = fit_pos,
                                        fit_fr = fit_fr,
                                        initial = initial,
                                        minimizer = 'leastsq',
                                        plot_science = False
                                        )
            else:
                res = data.fit_stars(ra_list,
                                        de_list,
                                        fr_list,
                                        fit_pos = fit_pos,
                                        fit_fr = fit_fr,
                                        initial = initial,
                                        minimizer = 'emcee',
                                        nthreads = int(self.input_dict["nthreads"]),
                                        nwalkers = int(self.input_dict["nwalkers"]),
                                        nsteps = int(self.input_dict["nsteps"]),
                                        plot_science = False
                                      )
            # self.res.append(data)
            self.update_progress.emit(fdx)
        self.finished.emit()


class PlotData(FigureCanvas):
    def __init__(self, parent=None, dpi=150):
        self.fig = Figure(dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.data_mapping = {
                    "Vis Amp": 0,
                    "Vis 2": 1,
                    "Closure": 2,
                    "Vis Phi": 3,
                }

    def plot_data(self, quant, data, lowest_plot=True, pol_idx=0):
        if quant not in ["Vis Amp", "Vis 2", "Closure", "Vis Phi"]:
            raise ValueError('quant for fitting function is wrong')
        logging.debug(f'Plotting {quant}')
        magu_as = np.copy(data.spFrequAS)
        magu_as_T3 = np.copy(data.spFrequAS_T3)

        bl_sort = [2, 3, 5, 0, 4, 1]
        cl_sort = [0, 3, 2, 1]
        nchannel = len(magu_as[0])
        for bl in range(6):
            magu_as[bl] = (np.linspace(nchannel, 0, nchannel)
                            + bl_sort[bl]*(nchannel+nchannel//2))
        for cl in range(4):
            magu_as_T3[cl] = (np.linspace(nchannel, 0, nchannel)
                                + cl_sort[cl]*(nchannel+nchannel//2))
        
        qdx = self.data_mapping[quant]
        dat = data.int_data[0][qdx*2 + pol_idx]
        err = data.int_data[1][qdx*2 + pol_idx]
        flag = data.int_data[2][qdx*2 + pol_idx]

        plot_fit = True
        try:
            plotdata = data.plotdata
        except AttributeError:
            plot_fit = False
            logging.warning('No fit data to plot')

        if plot_fit:
            theta, fitdata, fitarg, fithelp = plotdata[pol_idx]
            (nsource, fit_for, bispec_ind, fit_mode, wave, dlambda,
             fixedBHalpha, todel, fixed, phasemaps, northA, dra, ddec, amp_map_int,
             pha_map_int, amp_map_denom_int, fit_phasemaps, fix_pm_sources,
             fix_pm_amp_c, fix_pm_pha_c, fix_pm_int_c) = fithelp

            for ddx in range(len(todel)):
                theta = np.insert(theta, todel[ddx], fixed[ddx])

            wave_model = data.wlSC
            dlambda_model = data.dlambda
            fithelp[4] = wave_model
            fithelp[5] = dlambda_model
            self.wave = wave_model
            self.dlambda = dlambda_model
            fitres = _calc_vis_mstars(theta, fitarg, fithelp)
            _model = [fitres[0], fitres[0]**2, fitres[2], fitres[1]]
            model = _model[qdx]

        if qdx > 1:
            dat1 = data.int_data[0][qdx*2]
            dat2 = data.int_data[0][qdx*2 + 1]
            cmax = np.nanmax(np.abs(np.concatenate((dat1, dat2))))
            if cmax < 5:
                cmax = 10
            elif cmax < 100:
                cmax = cmax*1.5
            else:
                cmax = 180
            textpos = -cmax*1.1
            linepos = 0
        else:
            textpos = -0.1
            linepos = 1
        
        if qdx == 2:
            bls = 4
            x = magu_as_T3
            colors = data.colors_closure
            labels = data.closure_labels
        else:
            bls = 6
            x = magu_as
            colors = data.colors_baseline
            labels = data.baseline_labels

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        for i in range(bls):
            ax.errorbar(x[i, :], dat[i, :]*(1-flag)[i],
                        err[i, :]*(1-flag)[i],
                        color=colors[i],
                        ls='', lw=1, alpha=0.5, capsize=0)
            ax.scatter(x[i, :], dat[i, :]*(1-flag)[i],
                        color=colors[i], alpha=0.5)
            if plot_fit:
                ax.plot(x[i, :], model[i, :],
                         color='grey', zorder=100)

            if lowest_plot:
                ax.text(x[i, :].mean(), textpos,
                        labels[i], color=colors[i],
                        ha='center', va='center')
        ax.axhline(y=linepos, color='grey', ls='--', lw=1)
        ax.set_ylabel(quant)
        if qdx < 2:
            ax.set_ylim(-0.03, 1.1)
        else:
            ax.set_ylim(-cmax, cmax)
        ax.set_xticks([])
        if lowest_plot:
            self.fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.1)
        else:
            self.fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.05)
        self.draw()
