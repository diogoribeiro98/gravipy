import sys
import numpy as np
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow,
                                QPushButton,
                                QFileDialog, QLineEdit,
                                QComboBox, QLabel,
                                QTextEdit, 
                                QVBoxLayout, QHBoxLayout,
                                QWidget, QCheckBox, QGridLayout,
                                QProgressBar,
                                )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont
except ImportError:
    from PyQt5.QtWidgets import (QApplication, QMainWindow,
                                QPushButton,
                                QFileDialog, QLineEdit,
                                QComboBox, QLabel,
                                QTextEdit, 
                                QVBoxLayout, QHBoxLayout,
                                QWidget, QCheckBox, QGridLayout,
                                QProgressBar
                                )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont

from gui_utils import PlotData, LoggingHandler, LoadDataList, FitWorker, LoadFiles
import logging
from astropy.io import fits
import time


class GRAVITYfitGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle("GRAVITY multi source fitting")
        self.setGeometry(100, 100, 1700, 1000)
        
        # Top line of right panel
        # Create a button widget to open a file dialog
        button_layout = QHBoxLayout()
        button_load = QPushButton("Open Files", self)
        button_load.setMaximumSize(100, 30)
        button_load.clicked.connect(self.showFileDialog)
        button_layout.addWidget(button_load)

        # Create a combo box to select the OFFX/OFFY to use
        self.data_select = QLabel("Chose files with OFFX/OFFY:", self)
        self.data_select_combo = QComboBox(self)
        self.data_select_combo.setFixedSize(150, 30)        
        self.data_select_combo.currentIndexChanged.connect(self.updateFileList)
        button_layout.addWidget(self.data_select)
        button_layout.addWidget(self.data_select_combo)

        button_fit_layout = QHBoxLayout()
        button_fit_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # Button to show field
        button_show = QPushButton("Show field", self)
        button_show.setFixedSize(100, 30)
        button_show.clicked.connect(lambda: self.loadguesses(plot=True))
        button_fit_layout.addWidget(button_show)

        # Button to get guesses from header
        button_guess = QPushButton("Get fitting guess from header", self)
        button_guess.setFixedSize(250, 30)
        button_guess.clicked.connect(self.loadguesses)
        button_fit_layout.addWidget(button_guess)

        # Button to run the fit
        button_fit = QPushButton("Fit", self)
        button_fit.setFixedSize(100, 30)
        button_fit.clicked.connect(self.fit)
        button_fit_layout.addWidget(button_fit)

        # Layout to choose between files
        file_layout = QHBoxLayout()
        prev_button = QPushButton("⬅️")
        next_button = QPushButton("➡️")
        prev_button.setFixedSize(30, 30)
        next_button.setFixedSize(30, 30)
        prev_button.clicked.connect(self.prev_file)
        next_button.clicked.connect(self.next_file)
        file_layout.addWidget(prev_button)
        file_layout.addWidget(next_button)
        self.current_index = 0
        self.file_label = QLabel('Selected File')
        file_layout.addWidget(self.file_label)

        # Create a combo box (dropdown) to select the data source
        self.plotlabel = QLabel("Plot Quantity:", self)
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.setMaximumSize(100, 30)

        self.data_source_combo.addItems(["Vis Amp", "Vis 2", "Closure", "Vis Phi"])
        self.data_source_combo.currentIndexChanged.connect(self.updatePlot)

        self.canvas1 = PlotData(self)
        self.canvas2 = PlotData(self)
        
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("", self)
        progress_layout.addWidget(self.progress_label,
                                  alignment=Qt.AlignmentFlag.AlignRight)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedSize(150, 20)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        # Create a QTextEdit widget to display log messages
        self.log_text_edit = QTextEdit(self)
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setPlaceholderText("Log Messages")
        self.log_text_edit.setMinimumSize(500,150)
        self.log_handler = LoggingHandler(self.log_text_edit)
        self.log_handler.setFormatter(logging.Formatter('%(levelname)s: %(name)s - %(message)s'))
        self.log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

        main_layout = QGridLayout()
        right_widget = QWidget(self)
        right_layout = QVBoxLayout()
        right_layout.addLayout(file_layout)
        right_layout.addLayout(button_fit_layout)
        right_layout.addWidget(self.plotlabel)
        right_layout.addWidget(self.data_source_combo)
        right_layout.addWidget(self.canvas1)
        right_layout.addWidget(self.canvas2)
        right_layout.addWidget(self.log_text_edit)

        self.left_widget = QWidget(self)
        self.left_layout = QGridLayout()
        self.left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.createSourceLayout()

        main_layout.addLayout(button_layout, 0, 1,
                              alignment=Qt.AlignmentFlag.AlignLeft)
        main_layout.addLayout(right_layout, 2, 1)
        main_layout.addLayout(self.left_layout, 2, 0,
                              alignment=Qt.AlignmentFlag.AlignTop)
        main_layout.addLayout(progress_layout, 3, 1,
                              alignment=Qt.AlignmentFlag.AlignRight)

        # Create a central widget and set the main layout
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def createSourceLayout(self):
        try:
            for i in reversed(range(self.input_layout.count())):
                widget = self.input_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()
        except AttributeError:
            pass
        input_layout = QGridLayout() 

        fitlabel = QLabel("Fitting options:", self)
        fitlabel.setMaximumSize(200, 30)
        font = QFont()
        font.setPointSize(16)  # Set the font size to 16
        font.setBold(True)    # Make the font bold
        fitlabel.setFont(font)        
        input_layout.addWidget(fitlabel, 0, 0)


        source_label = QLabel('Number of sources')
        source_label.setMaximumSize(200, 30)
        source_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        source_label.setContentsMargins(0, 5, 0, 0)
        input_layout.addWidget(source_label, 2, 0)

        source_combo = QComboBox(self)
        source_combo.setMaximumSize(150, 30)
        source_combo.addItems(["2", "3", "4", "5"])
        try:
            source_combo.setCurrentText(str(self.nsources))
        except AttributeError:
            source_combo.setCurrentText("3")
        input_layout.addWidget(source_combo, 2, 1)
        source_combo.currentIndexChanged.connect(self.updateInputFields)
        self.source_combo = source_combo

        self.nsources = int(source_combo.currentText())
        try:
            if self.nsources != self.nsources_prev:
                logging.info(f"Number of sources: {self.nsources}")
        except AttributeError:
            pass
        self.nsources_prev = self.nsources
        for ndx in range(self.nsources-1):
            star_label = QLabel(f"Source {ndx+2}")
            star_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            star_label.setMaximumSize(100, 30)
            input_layout.addWidget(star_label, 4, ndx+1)

        try:
            self.input_dict
        except AttributeError:
            self.input_dict = {}
        self.input_mapping = {}
        input_labels = ["Sep. RA", "Sep. Dec", "Flux Ratio"]
        input_labels_dic = ["RA", "Dec", "fr"]
        input_init_val = ["0.0", "0.0", "1.0"]
        nsources = self.nsources
        for ldx, label_text in enumerate(input_labels):
            input_label = QLabel(label_text)
            input_label.setMaximumSize(200, 30)
            input_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            input_label.setContentsMargins(0, 5, 0, 0)
            input_layout.addWidget(input_label, ldx+5, 0)
            for i in range(nsources-1):
                if i == 0 and ldx == 2:
                    continue
                input_box = QLineEdit(self)
                self.input_mapping[input_box] = f"{input_labels_dic[ldx]} {i+1}"
                input_box.textChanged.connect(self.updateDictionary)
                if f'{input_labels_dic[ldx]} {i+1}' not in self.input_dict:
                    self.input_dict[f"{input_labels_dic[ldx]} {i+1}"] = f"{input_init_val[ldx]}"
                input_box.setText(self.input_dict[f"{input_labels_dic[ldx]} {i+1}"] )
                input_box.setMaximumSize(100, 30)
                input_layout.addWidget(input_box, ldx+5, i+1)

        try:
            self.checkbox_dict
        except AttributeError:
            self.checkbox_dict = {}
        self.checkbox_mapping = {}

        fit_labels = ["Fit Position", "Fit Flux Ratio"]
        fit_labels_dic = ["pos", "fr"]
        for ldx, label_text in enumerate(fit_labels):
            fit_label = QLabel(label_text)
            fit_label.setMaximumSize(200, 30)
            fit_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            fit_label.setContentsMargins(0, 5, 0, 0)
            input_layout.addWidget(fit_label, ldx+8, 0)
            for i in range(nsources-1):
                if i == 0 and ldx == 1:
                    continue
                if nsources == 2 and i == 1:
                    continue
                checkbox = QCheckBox()
                checkbox.setFixedSize(100, 30)
                self.checkbox_mapping[checkbox] = f"{fit_labels_dic[ldx]} {i+1}"
                checkbox.stateChanged.connect(self.checkboxStateChanged)
                if f'{fit_labels_dic[ldx]} {i+1}' not in self.checkbox_dict:
                    self.checkbox_dict[f"{fit_labels_dic[ldx]} {i+1}"] = True
                checkbox.setChecked(self.checkbox_dict[f"{fit_labels_dic[ldx]} {i+1}"])
                input_layout.addWidget(checkbox, ldx+8, i+1)
        
        init_labels = ["pc RA (Source 1 pos)", "pc Dec (Source 1 pos)",
                       "Power law index (Sou. 1)",
                       "fr Source 1 / Source 2", "Flux BG"]
        init_labels_dic = ["pcRA", "pcDec", "alphaBH", "frBH", "frBG"]
        init_init_val = ["0.0", "0.0", "3.0", "1.0", "1.0"]
        for ldx, label_text in enumerate(init_labels):
            init_label = QLabel(label_text)
            init_label.setMaximumSize(200, 30)
            init_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            init_label.setContentsMargins(0, 5, 0, 0)
            input_layout.addWidget(init_label, ldx+10, 0)
            init_box = QLineEdit(self)
            init_box.setMaximumSize(100, 30)
            self.input_mapping[init_box] = f"{init_labels_dic[ldx]}"
            init_box.textChanged.connect(self.updateDictionary)
            if f'{init_labels_dic[ldx]}' not in self.input_dict:
                self.input_dict[f"{init_labels_dic[ldx]}"] = f"{init_init_val[ldx]}""" 
            init_box.setText(self.input_dict[f"{init_labels_dic[ldx]}"] )
            input_layout.addWidget(init_box, ldx+10, 1)


        mode_label = QLabel('Fitting Mode')
        mode_label.setMaximumSize(200, 30)
        mode_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        mode_label.setContentsMargins(0, 5, 0, 0)
        input_layout.addWidget(mode_label, 15, 0)

        mode_combo = QComboBox(self)
        mode_combo.setMaximumSize(150, 30)
        mode_combo.addItems(["Least Sqr", "MCMC"])
        try:
            mode_combo.setCurrentText(str(self.minimizer))
        except AttributeError:
            mode_combo.setCurrentIndex(0)
        self.minimizer = mode_combo.currentText()
        input_layout.addWidget(mode_combo, 15, 1)
        mode_combo.currentIndexChanged.connect(self.updateInputFields)
        self.mode_combo = mode_combo

        if self.minimizer == "MCMC":
            mcmc_labels = ["nthreads", "nwalkers", "nsteps"]
            mcmc_init_val = ["4", "300", "300"]
            for ldx, label_text in enumerate(mcmc_labels):
                mcmc_label = QLabel(label_text)
                mcmc_label.setMaximumSize(200, 30)
                mcmc_label.setAlignment(Qt.AlignmentFlag.AlignRight)
                mcmc_label.setContentsMargins(0, 5, 0, 0)
                input_layout.addWidget(mcmc_label, ldx+16, 0)
                mcmc_box = QLineEdit(self)
                mcmc_box.setMaximumSize(100, 30)
                self.input_mapping[mcmc_box] = f"{label_text}"
                mcmc_box.textChanged.connect(self.updateDictionary)
                if f'{label_text}' not in self.input_dict:
                    self.input_dict[f"{label_text}"] = f"{mcmc_init_val[ldx]}"
                mcmc_box.setText(self.input_dict[f"{label_text}"] )
                input_layout.addWidget(mcmc_box, ldx+16, 1)

        self.input_layout = input_layout
        self.left_layout.addLayout(self.input_layout, 0, 0)

    def updateInputFields(self):
        self.nsources = int(self.source_combo.currentText())
        self.minimizer = self.mode_combo.currentText()
        self.createSourceLayout()
        
    def showFileDialog(self):
        # Create a file dialog and set its properties
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Files")
        # Show the file dialog and get the selected file(s)
        file_paths, test = file_dialog.getOpenFileNames(self, "Open File", "", "Fits files (*.fits);;All Files (*)")

        if not file_paths:
            return

        self.progress_bar.setMaximum(len(file_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setText(f"Loading files: ")

        self.loader = LoadFiles(file_paths)
        self.loader.update_progress.connect(self.update_progress)
        self.loader.finished.connect(self.showFileDialog_save)
        self.loader.start()

    def showFileDialog_save(self):
        self.files = self.loader.files
        self.offs = self.loader.offs
        self.offs = list(set(self.offs))
        offs_str = [str(i)[1:-1] for i in self.offs]
        self.data_select_combo.clear()
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.data_select_combo.addItems(offs_str)



    def update_progress(self, value):
        current_value = self.progress_bar.value() + 1
        self.progress_bar.setValue(current_value)

    def updateFileList(self):
        # get index of data_select_combo
        selectec_off = self.data_select_combo.currentIndex()
        _off = self.offs[selectec_off]
        self.sel_files = []
        self.sel_files_names = []
        for file in self.files:
            h = fits.open(file)[0].header
            if (h['ESO INS SOBJ OFFX'], h['ESO INS SOBJ OFFY']) == _off:
                self.sel_files.append(file)
                self.sel_files_names.append(file[file.rfind('GRAVI'):])
        self.len_sel_files = len(self.sel_files)
        self.current_index = 0
        
        self.progress_bar.setMaximum(self.len_sel_files)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setText(f"Loading files: ")            

        self.loader_data = LoadDataList(self.sel_files)
        self.loader_data.update_progress.connect(self.update_progress)
        self.loader_data.start()
        self.loader_data.finished.connect(self.updateFileList_save)


    def updateFileList_save(self):
        logging.info("Files loaded")
        self.data = self.loader_data.data
        self.polmode = self.data[0].polmode
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.file_label.setText(f'Selected File '
                                f'{self.current_index+1}/{self.len_sel_files}: '
                                f'{self.sel_files_names[self.current_index]}')
        self.updatePlot()


    def prev_file(self):
        try:
            self.sel_files
        except AttributeError:
            logging.error("No data loaded")
            return 0
        if self.current_index > 0:
            self.current_index -= 1
            self.update_sel_file()
            self.updatePlot()

    def next_file(self):
        try:
            self.sel_files
        except AttributeError:
            logging.error("No data loaded")
            return 0
        if self.current_index < self.len_sel_files - 1:
            self.current_index += 1
            self.update_sel_file()
            self.updatePlot()

    def update_sel_file(self):
        self.file_label.setText(f'Selected File '
                                f'{self.current_index+1}/{self.len_sel_files}: '
                                f'{self.sel_files_names[self.current_index]}')

    def updatePlot(self):
        selected_data_source = self.data_source_combo.currentText()
        data = self.data[self.current_index]
        try:
            logging.info(f"Selected data source: {selected_data_source}")
            if data.polmode == 'SPLIT':
                self.canvas1.plot_data(selected_data_source, data, lowest_plot=False, pol_idx=0)
                self.canvas2.plot_data(selected_data_source, data, lowest_plot=True, pol_idx=1)
            else:
                self.canvas1.plot_data(selected_data_source, data, lowest_plot=True, pol_idx=0)
        except Exception as e:
            logging.error(f"Error updating plot: {str(e)}")

    def loadguesses(self, plot=False):
        try:
            self.mfit = self.data[self.current_index]
        except AttributeError:
            logging.error("Cannot load guess, no data loaded")
            return 0
        try:
            ra_list, de_list, fr_list, initial = self.mfit.prep_fit(plot=plot)
        except ValueError:
            logging.error("Cannot load guess, only one source in field")
            return 0
        if plot:
            return 0
        
        self.nsources = len(ra_list) + 1
        for idx in range(self.nsources-1):
            self.input_dict[f"RA {idx+1}"] = f'{ra_list[idx]:.3f}'
            self.input_dict[f"Dec {idx+1}"] = f'{de_list[idx]:.3f}'
            if idx > 0:
                self.input_dict[f"fr {idx+1}"] = f'{fr_list[idx-1]:.3f}'
        self.input_dict["alphaBH"] = f'{initial[0]:.3f}'
        self.input_dict["frBG"] = f'{initial[1]:.3f}'
        self.input_dict["pcRA"] = f'{initial[2]:.3f}'
        self.input_dict["pcDec"] = f'{initial[3]:.3f}'
        self.input_dict["frBH"] = f'{initial[4]:.3f}'
        self.createSourceLayout()
        
    def fit(self):
        try:
            self.data
        except AttributeError:
            logging.error("Cannot fit, no data loaded")
            return 0

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
        
        logging.warning('Fit is running')
        logging.debug('Fitting input')
        logging.debug(f'ra list: {ra_list}')
        logging.debug(f'de list: {de_list}')
        logging.debug(f'fr list: {fr_list}')
        logging.debug(f'fit pos: {fit_pos}')
        logging.debug(f'fit fr: {fit_fr}')
        logging.debug(f'initial: {initial}')

        self.progress_bar.setMaximum(self.len_sel_files)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setText(f"Fitting files: ")
        self.loader_fit = FitWorker(self.data, self.input_dict, self.checkbox_dict,
                                    self.minimizer, self.nsources)
        self.loader_fit.update_progress.connect(self.update_progress)
        self.loader_fit.start()
        self.loader_fit.finished.connect(self.fit_save)


        # for _idx, _ind_data in enumerate(self.data):
        #     logging.info(f'Fitting file {_idx+1}/{self.len_sel_files}')
        #     self.worker = FitWorker(_ind_data.data, self.input_dict, self.checkbox_dict,
        #                             self.minimizer, self.nsources)
        #     self.worker.moveToThread(self.thread)
        #     self.thread.started.connect(self.worker.run)

        #     self.worker.finished.connect(self.thread.quit)
        #     self.worker.finished.connect(self.worker.deleteLater)
        #     self.thread.finished.connect(self.thread.deleteLater)

        #     self.thread.start()
        #     self.worker.run()

        # self.updatePlot()

     

    #     self.loader_data = LoadDataList(self.sel_files)
    #     self.loader_data.update_progress.connect(self.update_progress)
    #     self.loader_data.start()
    #     self.loader_data.finished.connect(self.updateFileList_save)


    def fit_save(self):
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        time.sleep(1)
        self.updatePlot()





    def checkboxStateChanged(self, state):
        sender = self.sender()
        if sender in self.checkbox_mapping:
            key = self.checkbox_mapping[sender]
            checked = state == 2  # 2 corresponds to checked state in Qt
            self.checkbox_dict[key] = checked
            logging.debug(f"Checkbox {key} state changed to {checked}")

    def updateDictionary(self, new_text):
        sender = self.sender()
        if sender in self.input_mapping:
            key = self.input_mapping[sender]
            self.input_dict[key] = new_text
            logging.debug(f"{key} value changed to {new_text}")

def main():
    app = QApplication(sys.argv)
    window = GRAVITYfitGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
