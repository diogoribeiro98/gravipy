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
                                QProgressBar, QSpacerItem,
                                QTableWidgetItem, QTableWidget
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
                                QProgressBar, QSpacerItem
                                )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont

from gui_utils import (PlotData, LoggingHandler,
                       LoadDataList, FitWorker,
                       LoadFiles, PlotResults)
import logging
from astropy.io import fits
import time

MAIN_COLOR = '#a1d99b'
DEFAULT_STYLE = """
QProgressBar{
    border: 2px solid grey;
    border-radius: 5px;
    text-align: center
}

QProgressBar::chunk {
    background-color: #a1d99b;
    width: 10px;
    margin: 0px;
}
"""

class GRAVITYfitGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.color_clicked = MAIN_COLOR
        self.plot_window = None

        # Set window properties
        self.setWindowTitle("GRAVITY multi source fitting")
        self.setGeometry(100, 100, 1700, 1100)

        self.create_button_layout()
        self.create_right_layout()
        self.create_progress_layout()


        main_layout = QGridLayout()
        self.left_layout = QGridLayout()
        self.left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.create_source_layout()

        self.result_layout = QVBoxLayout()
        self.result_layout.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.create_result_layout()

        # # left part
        main_layout.addLayout(self.left_layout, 2, 0,
                              alignment=Qt.AlignmentFlag.AlignTop)
        main_layout.addLayout(self.result_layout, 3, 0,
                              alignment=Qt.AlignmentFlag.AlignBottom)
        spacer1 = QSpacerItem(50, 1)
        main_layout.addItem(spacer1, 4, 0)
        
        # Middle
        spacer2 = QSpacerItem(50, 1)
        main_layout.addItem(spacer2, 1, 1)
        
        # right part
        main_layout.addLayout(self.button_layout, 0, 2,
                              alignment=Qt.AlignmentFlag.AlignLeft)
        main_layout.addLayout(self.right_layout, 2, 2, 3, 2)
        main_layout.addLayout(self.progress_layout, 5, 2)

        # Create a central widget and set the main layout
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)



    def create_button_layout(self):
        # Top line of right panel
        self.button_layout = QHBoxLayout()
        button_load = QPushButton("Open Files", self)
        button_load.setMaximumSize(100, 30)
        button_load.clicked.connect(self.show_file_dialog)
        self.button_layout.addWidget(button_load)
        self.data_select = QLabel("   Chose files with OFFX/OFFY:", self)
        self.data_select_combo = QComboBox(self)
        self.data_select_combo.setFixedSize(150, 30)        
        self.data_select_combo.currentIndexChanged.connect(self.update_file_list)
        self.button_layout.addWidget(self.data_select)
        self.button_layout.addWidget(self.data_select_combo)


    def create_right_layout(self):
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

        # Layout for buttons to fit
        button_fit_layout = QHBoxLayout()
        button_fit_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        button_show = QPushButton("Show field", self)
        button_show.setFixedSize(100, 30)
        # button_show.clicked.connect(lambda: self.load_guesses(plot=True))
        button_fit_layout.addWidget(button_show)
        button_guess = QPushButton("Get fitting guess from header", self)
        button_guess.setFixedSize(250, 30)
        button_guess.clicked.connect(self.load_guesses)
        button_fit_layout.addWidget(button_guess)
        button_fit = QPushButton("Fit", self)
        button_fit.setFixedSize(100, 30)
        button_fit.clicked.connect(self.fit)
        button_fit_layout.addWidget(button_fit)

        # Plotting
        plotlabel = QLabel("Plot Quantity:", self)
        self.data_source = QHBoxLayout()
        self.data_source.setAlignment(Qt.AlignmentFlag.AlignLeft)
        prev_data = QPushButton("⬅️")
        next_data = QPushButton("➡️")
        prev_data.setFixedSize(30, 30)
        next_data.setFixedSize(30, 30)
        prev_data.clicked.connect(self.prev_data)
        next_data.clicked.connect(self.next_data)
        self.data_source.addWidget(prev_data)
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.setMaximumSize(100, 30)
        self.data_source_combo.addItems(["Vis Amp", "Vis 2", "Closure", "Vis Phi"])
        self.data_source_combo.currentIndexChanged.connect(self.update_plots)
        self.data_source.addWidget(self.data_source_combo)
        self.data_source.addWidget(next_data)
        self.canvas1 = PlotData(self)
        self.canvas2 = PlotData(self)

        # Log messages
        self.log_text_edit = QTextEdit(self)
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setPlaceholderText("Log Messages")
        self.log_text_edit.setMinimumSize(500,150)
        self.log_handler = LoggingHandler(self.log_text_edit)
        self.log_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

        # Put it together
        self.right_layout = QVBoxLayout()
        self.right_layout.addLayout(file_layout)
        self.right_layout.addLayout(button_fit_layout)
        self.right_layout.addWidget(plotlabel)
        self.right_layout.addLayout(self.data_source)
        self.right_layout.addWidget(self.canvas1)
        self.right_layout.addWidget(self.canvas2)
        self.right_layout.addWidget(self.log_text_edit)

    def create_progress_layout(self):
        self.progress_layout = QHBoxLayout()
        self.progress_label = QLabel("", self)
        self.progress_layout.addWidget(self.progress_label,
                                       alignment=Qt.AlignmentFlag.AlignRight)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedSize(150, 20)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(DEFAULT_STYLE)
        self.progress_layout.addWidget(self.progress_bar)

    def create_source_layout(self):
        try:
            for i in reversed(range(self.input_layout.count())):
                widget = self.input_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()
        except AttributeError:
            pass
        try:
            for i in reversed(range(self.save_layout.count())):
                widget = self.save_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()
        except AttributeError:
            pass

        input_layout = QGridLayout() 

        try:
            self.checkbox_dict
        except AttributeError:
            self.checkbox_dict = {}
        self.checkbox_mapping = {}

        fitlabel = QLabel("Fitting options", self)
        fitlabel.setMaximumSize(200, 30)
        font = QFont()
        font.setPointSize(16)  # Set the font size to 16
        font.setBold(True)    # Make the font bold
        fitlabel.setFont(font)        
        input_layout.addWidget(fitlabel, 0, 0)

        hdx = 2
        source_label = QLabel('Fit for')
        source_label.setMaximumSize(200, 30)
        source_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        source_label.setContentsMargins(0, 5, 0, 0)
        input_layout.addWidget(source_label, hdx, 0)

        fit_labels = ["Vis Amp", "Vis 2", "Closure", "Vis Phi"]
        self.fit_labels_dic = ["VisAmp", "Vis2", "VisClo", "VisPhi"]
        self.fit_label_buttons = []
        for ldx, fitfor in enumerate(fit_labels):
            checkbox_button = QPushButton(fitfor)
            checkbox_button.setCheckable(True)
            checkbox_button.setFixedSize(100, 30)
            self.checkbox_mapping[checkbox_button] = f"{self.fit_labels_dic[ldx]}"
            checkbox_button.clicked.connect(self.checkbox_state_changed)
            if f'{self.fit_labels_dic[ldx]}' not in self.checkbox_dict:
                self.checkbox_dict[f"{self.fit_labels_dic[ldx]}"] = True
            checkbox_button.setChecked(self.checkbox_dict[f"{self.fit_labels_dic[ldx]}"])
            if checkbox_button.isChecked():
                checkbox_button.setStyleSheet(f"background-color: {self.color_clicked};")
            else:
                checkbox_button.setStyleSheet("background-color: white;")
            input_layout.addWidget(checkbox_button, hdx, 1+ldx)
            self.fit_label_buttons.append(checkbox_button)

        hdx += 1
        source_label = QLabel('Number of sources')
        source_label.setMaximumSize(200, 30)
        source_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        source_label.setContentsMargins(0, 5, 0, 0)
        input_layout.addWidget(source_label, hdx, 0)

        source_combo = QComboBox(self)
        source_combo.setMaximumSize(150, 30)
        source_combo.addItems(["2", "3", "4", "5"])
        try:
            source_combo.setCurrentText(str(self.nsources))
        except AttributeError:
            source_combo.setCurrentText("3")
        input_layout.addWidget(source_combo, hdx, 1)
        source_combo.currentIndexChanged.connect(self.update_input_fields)
        self.source_combo = source_combo
        hdx += 1

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
            input_layout.addWidget(star_label, hdx, ndx+1)

        hdx += 1
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
            input_layout.addWidget(input_label, ldx+hdx, 0)
            for i in range(nsources-1):
                if i == 0 and ldx == 2:
                    continue
                input_box = QLineEdit(self)
                self.input_mapping[input_box] = f"{input_labels_dic[ldx]} {i+1}"
                input_box.textChanged.connect(self.update_dictionary)
                if f'{input_labels_dic[ldx]} {i+1}' not in self.input_dict:
                    self.input_dict[f"{input_labels_dic[ldx]} {i+1}"] = f"{input_init_val[ldx]}"
                input_box.setText(self.input_dict[f"{input_labels_dic[ldx]} {i+1}"] )
                input_box.setMaximumSize(100, 30)
                input_layout.addWidget(input_box, ldx+hdx, i+1)

        hdx += 3
        fit_labels = ["Fit Position", "Fit Flux Ratio"]
        fit_labels_dic = ["pos", "fr"]
        for ldx, label_text in enumerate(fit_labels):
            fit_label = QLabel(label_text)
            fit_label.setMaximumSize(200, 30)
            fit_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            fit_label.setContentsMargins(0, 5, 0, 0)
            input_layout.addWidget(fit_label, ldx+hdx, 0)
            for i in range(nsources-1):
                if i == 0 and ldx == 1:
                    continue
                if nsources == 2 and i == 1:
                    continue
                checkbox = QCheckBox()
                checkbox.setFixedSize(100, 30)
                self.checkbox_mapping[checkbox] = f"{fit_labels_dic[ldx]} {i+1}"
                checkbox.stateChanged.connect(self.checkbox_state_changed)
                if f'{fit_labels_dic[ldx]} {i+1}' not in self.checkbox_dict:
                    self.checkbox_dict[f"{fit_labels_dic[ldx]} {i+1}"] = True
                checkbox.setChecked(self.checkbox_dict[f"{fit_labels_dic[ldx]} {i+1}"])
                input_layout.addWidget(checkbox, ldx+hdx, i+1)
        
        hdx += 2 
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
            input_layout.addWidget(init_label, ldx+hdx, 0)
            init_box = QLineEdit(self)
            init_box.setMaximumSize(100, 30)
            self.input_mapping[init_box] = f"{init_labels_dic[ldx]}"
            init_box.textChanged.connect(self.update_dictionary)
            if f'{init_labels_dic[ldx]}' not in self.input_dict:
                self.input_dict[f"{init_labels_dic[ldx]}"] = f"{init_init_val[ldx]}""" 
            init_box.setText(self.input_dict[f"{init_labels_dic[ldx]}"] )
            input_layout.addWidget(init_box, ldx+hdx, 1)

        hdx += 5
        mode_label = QLabel('Fitting Mode')
        mode_label.setMaximumSize(200, 30)
        mode_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        mode_label.setContentsMargins(0, 5, 0, 0)
        input_layout.addWidget(mode_label, hdx, 0)

        mode_combo = QComboBox(self)
        mode_combo.setMaximumSize(150, 30)
        mode_combo.addItems(["Least Sqr", "MCMC"])
        try:
            mode_combo.setCurrentText(str(self.minimizer))
        except AttributeError:
            mode_combo.setCurrentIndex(0)
        self.minimizer = mode_combo.currentText()
        input_layout.addWidget(mode_combo, hdx, 1)
        mode_combo.currentIndexChanged.connect(self.update_input_fields)
        self.mode_combo = mode_combo

        hdx += 1
        if self.minimizer == "MCMC":
            mcmc_labels = ["nthreads", "nwalkers", "nsteps"]
            mcmc_init_val = ["4", "300", "300"]
            for ldx, label_text in enumerate(mcmc_labels):
                mcmc_label = QLabel(label_text)
                mcmc_label.setMaximumSize(100, 30)
                mcmc_label.setAlignment(Qt.AlignmentFlag.AlignRight)
                mcmc_label.setContentsMargins(0, 5, 0, 0)
                input_layout.addWidget(mcmc_label, ldx+hdx-3, 2)
                mcmc_box = QLineEdit(self)
                mcmc_box.setMaximumSize(100, 30)
                self.input_mapping[mcmc_box] = f"{label_text}"
                mcmc_box.textChanged.connect(self.update_dictionary)
                if f'{label_text}' not in self.input_dict:
                    self.input_dict[f"{label_text}"] = f"{mcmc_init_val[ldx]}"
                mcmc_box.setText(self.input_dict[f"{label_text}"] )
                input_layout.addWidget(mcmc_box, ldx+hdx-3, 3)
        
        self.save_layout = QHBoxLayout()
        if self.minimizer == "MCMC":
            folder_button = QPushButton('I/O folder for MCMC')
            # folder_button.clicked.connect(self.open_folder_dialog)
            folder_button.setFixedSize(160, 30)
            self.save_layout.addWidget(folder_button)
            self.folder_text = QLineEdit()
            self.folder_text.setReadOnly(True)
            self.folder_text.setMaximumSize(420, 30)
            try:
                self.folder_text.setText(self.save_folder)
            except AttributeError:
                self.folder_text.setText('Pick folder')
            self.save_layout.addWidget(self.folder_text)

        self.input_layout = input_layout
        self.left_layout.addLayout(self.input_layout, 0, 0)
        self.left_layout.addLayout(self.save_layout, 1, 0)

    def create_result_layout(self):
        try:
            for i in reversed(range(self.fitheader.count())):
                widget = self.fitheader.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()
        except AttributeError:
            pass
        try:
            for i in reversed(range(self.result_layout.count())):
                widget = self.result_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()
        except AttributeError:
            pass
        try:
            data = self.data[self.current_index]
        except AttributeError:
            return 0
        try:
            fitres = data.fittab.copy()
        except (AttributeError, UnboundLocalError):
            return 0
        
        self.fitheader = QHBoxLayout()
        self.fitheader.setAlignment(Qt.AlignmentFlag.AlignBottom)
        fitlabel = QLabel("Fitting results   ", self)
        fitlabel.setMaximumSize(200, 30)
        font = QFont()
        font.setPointSize(16)  # Set the font size to 16
        font.setBold(True)    # Make the font bold
        fitlabel.setFont(font)
        self.fitheader.addWidget(fitlabel)

        button_show = QPushButton("Plot fit results", self)
        button_show.setFixedSize(150, 30)
        button_show.clicked.connect(self.plot_results)
        self.fitheader.addWidget(button_show)
        self.result_layout.addLayout(self.fitheader)

        table_widget = QTableWidget()
        table_widget.setStyleSheet("border: none;")
        table_widget.setColumnCount(3)
        table_widget.setHorizontalHeaderLabels(['', 'P1', 'P2'])
        table_widget.setColumnWidth(0, 190)
        table_widget.setColumnWidth(1, 100)
        table_widget.setColumnWidth(2, 100)
        
        table_text = []
        for sdx in range(1,self.nsources):
            table_text.append([f'Sep. RA {sdx-1}:', f'{fitres[f"dRA{sdx}"][1]:.3f}', f'{fitres[f"dRA{sdx}"][6]:.3f}'])
            table_text.append([f'Sep. Dec {sdx-1}:', f'{fitres[f"dDEC{sdx}"][1]:.3f}', f'{fitres[f"dDEC{sdx}"][6]:.3f}'])
            if sdx > 1:
                table_text.append([f'Flux Ratio {sdx-1}:', f'{fitres[f"fr{sdx}"][1]:.3f}', f'{fitres[f"fr{sdx}"][6]:.3f}'])
        table_text.append([f'pc RA (Source 1 pos):', f'{fitres["pc_RA"][1]:.3f}', f'{fitres["pc_RA"][6]:.3f}'])
        table_text.append([f'pc Dec (Source 1 pos):', f'{fitres["pc_Dec"][1]:.3f}', f'{fitres["pc_Dec"][6]:.3f}'])
        table_text.append([f'Power law index:', f'{fitres["alpha_BH"][1]:.3f}', f'{fitres["alpha_BH"][6]:.3f}'])
        table_text.append([f'fr Source 1 / Source 2:', f'{fitres["fr_BH"][1]:.3f}', f'{fitres["fr_BH"][6]:.3f}'])
        table_text.append([f'Flux BG:', f'{fitres["f_BG"][1]:.3f}', f'{fitres["f_BG"][6]:.3f}'])

        table_widget.setRowCount(len(table_text)) 
        for row_index, row_data in enumerate(table_text):
            for col_index, cell_data in enumerate(row_data):
                item = QTableWidgetItem(cell_data)
                item.setTextAlignment(0x0082)
                table_widget.setItem(row_index, col_index, item)

        table_widget.verticalHeader().setVisible(False)
        # table_widget.horizontalHeader().setStretchLastSection(True)
        table_widget.setFixedWidth(410)
        table_widget.setMaximumHeight((len(table_text)+1)*30)
        self.result_layout.addWidget(table_widget)
        
    def show_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Files")
        file_paths, test = file_dialog.getOpenFileNames(self, "Open File", "", "Fits files (*.fits);;All Files (*)")
        if not file_paths:
            return

        self.progress_bar.setMaximum(len(file_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setText(f"Loading all files: ")

        self.loader = LoadFiles(file_paths)
        self.loader.update_progress.connect(self.update_progress)
        self.loader.finished.connect(self.show_file_dialog_save)
        self.loader.start()

    def show_file_dialog_save(self):
        self.files = self.loader.files
        self.offs = self.loader.offs
        self.offs = list(set(self.offs))
        offs_str = [str(i)[1:-1] for i in self.offs]
        self.data_select_combo.clear()
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.data_select_combo.addItems(offs_str)

    def open_folder_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select a Folder")
        folder_path = file_dialog.getExistingDirectory(self, "Select a Folder")
        if folder_path:
            self.folder_text.setText(folder_path)
            self.save_folder = folder_path
    
    def prev_file(self):
        try:
            self.sel_files
        except AttributeError:
            logging.error("No data loaded")
            return 0
        if self.current_index > 0:
            self.current_index -= 1
            self.update_sel_file()
            self.update_plots()

    def next_file(self):
        try:
            self.sel_files
        except AttributeError:
            logging.error("No data loaded")
            return 0
        if self.current_index < self.len_sel_files - 1:
            self.current_index += 1
            self.update_sel_file()
            self.update_plots()

    def prev_data(self):
        cur = self.data_source_combo.currentIndex()
        if cur > 0:
            self.data_source_combo.setCurrentIndex(cur-1)

    def next_data(self):
        cur = self.data_source_combo.currentIndex()
        if cur < self.data_source_combo.count()-1:
            self.data_source_combo.setCurrentIndex(cur+1)

    def update_progress(self, value):
        current_value = self.progress_bar.value() + 1
        self.progress_bar.setValue(current_value)

    def update_file_list(self):
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
        self.progress_label.setText(f"Loading files from field: ")            

        self.loader_data = LoadDataList(self.sel_files)
        self.loader_data.update_progress.connect(self.update_progress)
        self.loader_data.start()
        self.loader_data.finished.connect(self.update_file_list_save)

    def update_file_list_save(self):
        logging.info("Files loaded")
        self.data = self.loader_data.data
        self.polmode = self.data[0].polmode
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self.file_label.setText(f'Selected File '
                                f'{self.current_index+1}/{self.len_sel_files}: '
                                f'{self.sel_files_names[self.current_index]}')
        self.update_plots()

    def update_sel_file(self):
        self.file_label.setText(f'Selected File '
                                f'{self.current_index+1}/{self.len_sel_files}: '
                                f'{self.sel_files_names[self.current_index]}')

    def update_plots(self):
        selected_data_source = self.data_source_combo.currentText()
        try:
            data = self.data[self.current_index]
        except AttributeError:
            logging.error("Cannot plot, no data loaded")
            return 0
        try:
            logging.debug(f"Selected data source: {selected_data_source}")
            if data.polmode == 'SPLIT':
                self.canvas1.plot_data(selected_data_source, data, lowest_plot=False, pol_idx=0)
                self.canvas2.plot_data(selected_data_source, data, lowest_plot=True, pol_idx=1)
            else:
                self.canvas1.plot_data(selected_data_source, data, lowest_plot=True, pol_idx=0)
        except Exception as e:
            logging.error(f"Error updating plot: {str(e)}")
        self.create_result_layout()

    def update_input_fields(self):
        self.nsources = int(self.source_combo.currentText())
        self.minimizer = self.mode_combo.currentText()
        self.create_source_layout()
        
    def update_dictionary(self, new_text):
        sender = self.sender()
        if sender in self.input_mapping:
            key = self.input_mapping[sender]
            self.input_dict[key] = new_text
            logging.debug(f"{key} value changed to {new_text}")
            
    def load_guesses(self, plot=False):
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
        self.create_source_layout()

    def checkbox_state_changed(self, state):
        sender = self.sender()
        if sender in self.checkbox_mapping:
            key = self.checkbox_mapping[sender]
            if key in self.fit_labels_dic:
                checked = state
                pdx = self.fit_labels_dic.index(key)
                if checked:
                    self.fit_label_buttons[pdx].setStyleSheet(f"background-color: {self.color_clicked};")
                else:
                    self.fit_label_buttons[pdx].setStyleSheet("background-color: white;")
            else:
                checked = state == 2  # 2 corresponds to checked state in Qt
            self.checkbox_dict[key] = checked
            logging.debug(f"Checkbox {key} state changed to {checked}")
        
        fit_labels_dic = ["VisAmp", "Vis2", "VisClo", "VisPhi"]
        self.fit_for = np.multiply([self.checkbox_dict[f] for f in fit_labels_dic],1)

    def fit(self):
        try:
            self.data
        except AttributeError:
            logging.error("Cannot fit, no data loaded")
            return 0

        self.progress_bar.setMaximum(self.len_sel_files)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setText(f"Fitting files: ")
        self.loader_fit = FitWorker(self.data, self.input_dict, self.checkbox_dict,
                                    self.minimizer, self.nsources, self.fit_for)
        self.loader_fit.update_progress.connect(self.update_progress)
        self.loader_fit.start()
        self.loader_fit.finished.connect(self.fit_save)

    def fit_save(self):
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        time.sleep(1)
        self.update_plots()
        self.create_result_layout()
        logging.info('Fit done')

    def plot_results(self):
        logging.info('Plotting results')
        
        # for all elements of data, check if fittab exists and plot all fittab['dRA1']
        try:
            self.data
        except AttributeError:
            logging.error("Cannot plot, no data loaded")
            return 0
        allfitres = []
        for data in self.data:
            try:
                fitres = data.fittab.copy()
                allfitres.append(fitres)
            except (AttributeError, UnboundLocalError):
                continue
        if len(allfitres) == 0:
            logging.error("No fit results available")
            return 0

        if self.plot_window is not None:
            self.plot_window.close()
            self.plot_window = None
        fit_pos = [self.checkbox_dict[f'pos {i}']
                   for i in range(1, self.nsources)]
        nplot = sum(fit_pos) + 2
        self.plot_window = PlotResults(allfitres,
                                       self.input_dict,
                                       self.checkbox_dict,
                                       figsize=[nplot*2.5, 3]
                                       )
        self.plot_window.show()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GRAVITYfitGUI()
    window.show()
    app.exec()
    del app

