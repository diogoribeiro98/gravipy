import mygravipy as gp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import gridspec, colors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import logging
from mygravipy.gravmfit import _calc_vis_mstars
from mygravipy.gcorbits import GCorbits
from astropy.io import fits
from astropy.stats import mad_std as mad
import warnings
import matplotlib.cbook

# try:
#     from PyQt6.QtCore import QThread, pyqtSignal
#     from PyQt6.QtWidgets import (QVBoxLayout, QWidget,
#                                  QLabel, QMainWindow)
# except ImportError:
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QVBoxLayout, QWidget,
                                QLabel, QMainWindow)

MAIN_COLOR = '#a1d99b'
SECOND_COLOR = '#74c476'
THIRD_COLOR = '#31a354'
FOURTH_COLOR = '#006d2c'
FIFTH_COLOR = '#00441b'

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

    def __init__(self, data, input_dict, checkbox_dict,
                 minimizer, nsources, fit_for, save_folder):
        super().__init__()
        self.data = data
        self.input_dict = input_dict
        self.checkbox_dict = checkbox_dict
        self.minimizer = minimizer
        self.nsources = nsources
        self.fit_for = fit_for
        self.save_folder = save_folder

    def run(self):
        warnings.filterwarnings("ignore", category=UserWarning)
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
                   3,
                   3,
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
                                        fit_for = self.fit_for,
                                        plot_science = False,
                                        create_pdf = self.checkbox_dict['create_pdf'],
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
                                     fit_for = self.fit_for,
                                     plot_science = False,
                                     savemcmc = self.save_folder,
                                     refit = self.checkbox_dict['refit'],
                                     create_pdf = self.checkbox_dict['create_pdf'],
                                    )
            self.update_progress.emit(fdx)
        self.finished.emit()
        warnings.resetwarnings()


class PlotData(FigureCanvas):
    def __init__(self, parent=None, dpi=150, figsize=None):
        if figsize is None:
            self.fig = Figure(dpi=dpi, facecolor='#e9e9e9')
        else:
            self.fig = Figure(dpi=dpi, facecolor='#e9e9e9',
                              figsize=(figsize[0], figsize[1]))
        super().__init__(self.fig)
        self.setParent(parent)

        self.data_mapping = {
                    "Vis Amp": 0,
                    "Vis 2": 1,
                    "Closure": 2,
                    "Vis Phi": 3,
                }

    def plot_walker(self, w1, w2, names):
        ws = [w1, w2]
        dim = len(names)
        gs = gridspec.GridSpec(dim, 2,
                               wspace=0.1)
        for wdx, samples in enumerate(ws):
            for i in range(dim):
                ax = self.fig.add_subplot(gs[i, wdx])
                ax.plot(samples[:, :, i].T, "k", alpha=0.3)
                # ax.axhline(mostprop[i], color='C0', alpha=0.5)
                ax.yaxis.set_label_coords(-0.1, 0.5)
                if wdx == 0:
                    ax.set_ylabel(names[i], fontsize=6)
                if i == dim-1:
                    ax.set_xlabel("step number")
                else:
                    ax.set_xticks([])
                if i == 0:
                    ax.set_title(f'Polarization {wdx+1}', fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)
        self.draw()

    def plot_field(self, t, off, lim=400, fiberrad=70):
        plt.rcParams["font.family"] = "sans-serif"
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        orb = GCorbits(t=t, loglevel='WARNING')
        stars, starpos = orb.find_stars(*off, plotlim=lim,
                                        fiberrad=fiberrad)

        for s in starpos:
            n, sx, sy, _, _ = s
            if np.any(np.abs(sx) > lim) or np.any(np.abs(sy) > lim):
                continue
            color = 'grey'
            for s in stars:
                if n == s[0]:
                    color = FOURTH_COLOR
            ax.scatter(sx, sy, c=color, s=7)
            ax.text(sx-3, sy, '%s' % (n), fontsize=6, color=color)
        ax.set_xlim(lim*1.2, -lim*1.2)
        ax.set_ylim([-lim*1.2, lim*1.2])
        circ = plt.Circle(off, radius=fiberrad, facecolor="None",
                        edgecolor=FOURTH_COLOR, linewidth=0.2)
        ax.add_artist(circ)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('RA [mas]', fontsize=8)
        ax.set_ylabel('Dec [mas]', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)
        self.draw()
        
    def plot_results(self, allfitres, dict_input, dict_checked):
        plt.rcParams["font.family"] = "sans-serif"
        _res = allfitres[0]
        # check how many sources were fitted
        nsources = 0
        for idx in range(1, 10):
            try:
                _res[f'dRA{idx}']
            except KeyError:
                break
            nsources += 1

        nplot = nsources + 2
        gs = gridspec.GridSpec(1, nplot,
                               wspace=0.2,
                               width_ratios=[1]*(nplot-1) + [0.05],)
        pcolors = np.linspace((1,1),
                             (len(allfitres),len(allfitres)),
                             len(allfitres))
        cmap = plt.cm.turbo
        norm = colors.BoundaryNorm(np.arange(0.5, len(allfitres)+1, 1),
                                   cmap.N)
        ndx = 0
        for idx in range(1, nsources+1):
            # if not fit_pos[idx-1]: continue
            dRA = [[r[f'dRA{idx}'][2],
                    r[f'dRA{idx}'][7]]  for r in allfitres]
            ddRA = [[(r[f'dRA{idx}'][3]+r[f'dRA{idx}'][4])/2,
                     (r[f'dRA{idx}'][8]+r[f'dRA{idx}'][9])/2]
                     for r in allfitres]
            dDEC = [[r[f'dDEC{idx}'][1],
                     r[f'dDEC{idx}'][6]]  for r in allfitres]
            ddDEC = [[(r[f'dDEC{idx}'][3]+r[f'dDEC{idx}'][4])/2,
                      (r[f'dDEC{idx}'][8]+r[f'dDEC{idx}'][9])/2]
                      for r in allfitres]
            mra1 = np.nanmedian(dRA)
            dra1 = mad(dRA)
            mde1 = np.nanmedian(dDEC)
            dde1 = mad(dDEC)


            ax = self.fig.add_subplot(gs[0, ndx])
            ax.set_facecolor("#e9e9e9")
            ax.scatter(dRA, dDEC, c=pcolors, cmap=cmap,
                       norm=norm, zorder=5)
            ax.errorbar(np.array(dRA).flatten(),
                        np.array(dDEC).flatten(),
                        np.array(ddDEC).flatten(),
                        np.array(ddRA).flatten(),
                        c='grey', ls='', lw=0.5,
                        capsize=0, zorder=1)
            ax.errorbar(mra1, mde1, dde1, dra1,
                        color='k', capsize=0, zorder=10,
                        label=f'Ra:  {mra1:.2f} +- {dra1:.2f}\n'
                            f'Dec: {mde1:.2f} +- {dde1:.2f}')
            try:
                gRa = float(dict_input[f'RA {idx}'])
                gDec = float(dict_input[f'Dec {idx}'])
                ax.axvline(gRa, ls='--', lw=0.5, color='grey', zorder=0)
                ax.axhline(gDec, ls='--', lw=0.5, color='grey', zorder=0)
            except KeyError:
                pass
            ax.set_aspect('equal', 'box')
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()
            x_range = max(x_limits) - min(x_limits)
            y_range = max(y_limits) - min(y_limits)

            max_range = max(x_range, y_range)
            if max_range < 0.5: max_range = 0.5
            ax.set_xlim(np.mean(x_limits) + max_range/2,
                        np.mean(x_limits) - max_range/2)
            ax.set_ylim(np.mean(y_limits) - max_range/2,
                        np.mean(y_limits) + max_range/2)
            ax.set_title(f'Separation Source {idx}', fontsize=8)
            ax.set_xlabel('RA [mas]', fontsize=8)
            if ndx == 0:
                ax.set_ylabel('Dec [mas]', fontsize=8)
            ax.legend(fontsize=6, loc=1, frameon=True, fancybox=True)
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.tick_params(axis='both', which='minor', labelsize=6)
            ndx += 1

        dRA = [[r[f'pc_RA'][2],
                r[f'pc_RA'][7]]  for r in allfitres]
        ddRA = [[(r[f'pc_RA'][3]+r[f'pc_RA'][4])/2,
                 (r[f'pc_RA'][8]+r[f'pc_RA'][9])/2]
                 for r in allfitres]
        dDEC = [[r[f'pc_Dec'][1],
                 r[f'pc_Dec'][6]]  for r in allfitres]
        ddDEC = [[(r[f'pc_Dec'][3]+r[f'pc_Dec'][4])/2,
                  (r[f'pc_Dec'][8]+r[f'pc_Dec'][9])/2]
                  for r in allfitres]
        mra1 = np.nanmedian(dRA)
        dra1 = mad(dRA)
        mde1 = np.nanmedian(dDEC)
        dde1 = mad(dDEC)

        gRa = float(dict_input['pcRA'])
        gDec = float(dict_input['pcDec'])

        ax = self.fig.add_subplot(gs[0, ndx])
        ax.set_facecolor("#e9e9e9")
        cbar = ax.scatter(dRA, dDEC, c=pcolors, cmap=cmap,
                          norm=norm, zorder=2)
        ax.errorbar(np.array(dRA).flatten(),
                    np.array(dDEC).flatten(),
                    np.array(ddDEC).flatten(),
                    np.array(ddRA).flatten(),
                    c='grey', ls='', lw=0.5,
                    capsize=0, zorder=1)

        ax.errorbar(mra1, mde1, dde1, dra1,
                    color='k', capsize=0, zorder=3,
                    label=f'Ra:  {mra1:.2f} +- {dra1:.2f}\n'
                        f'Dec: {mde1:.2f} +- {dde1:.2f}')
        ax.axvline(gRa, ls='--', lw=0.5, color='grey', zorder=0)
        ax.axhline(gDec, ls='--', lw=0.5, color='grey', zorder=0)
        ax.legend(fontsize=6, loc=1, frameon=True, fancybox=True)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)
        ax.set_aspect('equal', 'box')
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        x_range = max(x_limits) - min(x_limits)
        y_range = max(y_limits) - min(y_limits)
        max_range = max(x_range, y_range)
        if max_range < 0.5: max_range = 0.5
        ax.set_xlim(np.mean(x_limits) + max_range/2,
                    np.mean(x_limits) - max_range/2)
        ax.set_ylim(np.mean(y_limits) - max_range/2,
                    np.mean(y_limits) + max_range/2)
        ax.set_title('Central Source', fontsize=8)
        ax.set_xlabel('RA [mas]', fontsize=8)
        if ndx == 0:
            ax.set_ylabel('Dec [mas]', fontsize=8)
        ndx += 1

        cbaxes = self.fig.add_subplot(gs[0, ndx])
        cb = self.fig.colorbar(cbar, cax=cbaxes)
        cb.set_label('# File', fontsize=6)

        cb.set_ticks(np.arange(1, len(allfitres)+1, 1))
        cb.ax.tick_params(labelsize=6)
        self.draw()

    def plot_data(self, quant, data, lowest_plot=True, pol_idx=0):
        plt.rcParams["font.family"] = "sans-serif"
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
            fittab = data.fittab
            theta = fittab.iloc[[2+pol_idx*5]]
            nsources = 0
            for idx in range(1, 10):
                try:
                    theta[f'dRA{idx}']
                except KeyError:
                    break
                nsources += 1

        except AttributeError:
            plot_fit = False
        
        if plot_fit:
            try:
                data.dlambda
            except AttributeError:
                data.get_dlambda()
            
            try:
                data.pm_sources
                phasemap = True
            except AttributeError:
                phasemap = False
                logging.warning('No phase map data available')

            fithelp = [nsources, None, data.bispec_ind, 'numeric', data.wlSC, data.dlambda,
                       None, None, phasemap, None, None, None,
                       None, None, None,
                       False, data.pm_sources, False]

            wave_model = data.wlSC
            dlambda_model = data.dlambda
            fithelp[4] = wave_model
            fithelp[5] = dlambda_model
            self.wave = wave_model
            self.dlambda = dlambda_model
            fitres = _calc_vis_mstars(theta.values[0,1:-1],
                                      [data.u, data.v], fithelp)
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
        ax.set_facecolor("#e9e9e9")
        for i in range(bls):
            ax.errorbar(x[i, :], dat[i, :]*(1-flag)[i],
                        err[i, :]*(1-flag)[i],
                        color=FOURTH_COLOR,
                        ls='', lw=1, alpha=0.5, capsize=0)
            ax.scatter(x[i, :], dat[i, :]*(1-flag)[i],
                        color=FOURTH_COLOR, alpha=0.5)
            if plot_fit:
                ax.plot(x[i, :], model[i, :],
                         color='grey', zorder=100)

            if lowest_plot:
                ax.text(x[i, :].mean(), textpos,
                        labels[i], color='k',#colors[i],
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


class PlotResults(QMainWindow):
    def __init__(self, allres, input_dict, checked_dict, figsize=None):
        super().__init__()
        self.setStyleSheet("background-color: #e9e9e9;")
        layout = QVBoxLayout()
        self.label = QLabel("Fitting results")
        layout.addWidget(self.label)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        p = PlotData(self, dpi=160, figsize=figsize)
        layout.addWidget(p)
        p.plot_results(allres, input_dict, checked_dict)

class PlotStarPos(QMainWindow):
    def __init__(self, t, off):
        super().__init__()
        self.setStyleSheet("background-color: #e9e9e9;")
        layout = QVBoxLayout()
        self.setGeometry(200, 200, 700, 700)
        self.setWindowTitle("Expected star positions")

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        p = PlotData(self)
        layout.addWidget(p)
        p.plot_field(t, off) 

class PlotWalker(QMainWindow):
    def __init__(self, w1, w2, names):
        super().__init__()
        self.setStyleSheet("background-color: #e9e9e9;")
        layout = QVBoxLayout()
        # self.setGeometry(200, 200, 700, 700)
        self.setWindowTitle("MCMC walkers")

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        p = PlotData(self)
        layout.addWidget(p)
        p.plot_walker(w1, w2, names) 
