import mygravipy as gp
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import logging


class LoggingHandler(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        log_message = self.format(record)
        self.text_edit.append(log_message)


class LoadData:
    def __init__(self, filename):
        data = gp.GravData(filename)
        data.get_int_data()
        self.data = data
        self.polmode = data.polmode


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
