import sys
import numpy as np
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow,
                                QPushButton, QMessageBox,
                                QFileDialog, QLineEdit,
                                QComboBox, QLabel,
                                QTextEdit, QFormLayout,
                                QVBoxLayout, QHBoxLayout,
                                QWidget, QCheckBox, QGridLayout
                                )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont
except ImportError:
    from PyQt5.QtWidgets import (QApplication, QMainWindow,
                                QPushButton, QMessageBox,
                                QFileDialog, QLineEdit,
                                QComboBox, QLabel,
                                QTextEdit, QFormLayout,
                                QVBoxLayout, QHBoxLayout,
                                QWidget, QCheckBox, QGridLayout
                                )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
from gui_utils import PlotData, LoggingHandler, LoadData
import logging


class GRAVITYfitGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle("GRAVITY multi source fitting")
        self.setGeometry(100, 100, 1700, 1000)

        # Create a button to open the file dialog
        self.button = QPushButton("Open File", self)
        self.button.setMaximumSize(100, 30)
        self.button.clicked.connect(self.showFileDialog)

        # Create a QLineEdit widget to display the selected file path
        self.file_path_edit = QLineEdit(self)
        self.file_path_edit.setPlaceholderText("Selected File")

        # Create a combo box (dropdown) to select the data source
        self.plotlabel = QLabel("Plot Quantity:", self)
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.setMaximumSize(100, 30)

        self.data_source_combo.addItems(["Vis Amp", "Vis 2", "Closure", "Vis Phi"])
        self.data_source_combo.currentIndexChanged.connect(self.updatePlot)

        self.canvas1 = PlotData(self)
        self.canvas2 = PlotData(self)

        # Create a QTextEdit widget to display log messages
        self.log_text_edit = QTextEdit(self)
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setPlaceholderText("Log Messages")
        self.log_handler = LoggingHandler(self.log_text_edit)
        self.log_handler.setFormatter(logging.Formatter('%(levelname)s: %(name)s - %(message)s'))
        self.log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

        main_layout = QGridLayout()
        right_widget = QWidget(self)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.plotlabel)
        right_layout.addWidget(self.data_source_combo)
        right_layout.addWidget(self.canvas1)
        right_layout.addWidget(self.canvas2)
        right_layout.addWidget(self.log_text_edit)

        self.left_widget = QWidget(self)
        self.left_layout = QGridLayout()
        self.left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.createSourceLayout()

        main_layout.addWidget(self.button, 0, 0,
                              alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self.file_path_edit, 0, 1)
        main_layout.addLayout(right_layout, 1, 1)
        main_layout.addLayout(self.left_layout, 1, 0,
                              alignment=Qt.AlignmentFlag.AlignTop)

        # Create a central widget and set the main layout
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def updatePlot(self):
        selected_data_source = self.data_source_combo.currentText()
        try:
            logging.info(f"Selected data source: {selected_data_source}")
            if self.data.polmode == 'SPLIT':
                self.canvas1.plot_data(selected_data_source, self.data.data, lowest_plot=False, pol_idx=0)
                self.canvas2.plot_data(selected_data_source, self.data.data, lowest_plot=True, pol_idx=1)
            else:
                self.canvas1.plot_data(selected_data_source, self.data.data, lowest_plot=True, pol_idx=0)
        except Exception as e:
            logging.error(f"Error updating plot: {str(e)}")

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
        input_layout.addWidget(source_label, 1, 0)

        source_combo = QComboBox(self)
        source_combo.setMaximumSize(150, 30)
        source_combo.addItems(["2", "3", "4", "5"])
        try:
            source_combo.setCurrentText(str(self.nsoures))
        except AttributeError:
            source_combo.setCurrentText("3")
        input_layout.addWidget(source_combo, 1, 1)
        source_combo.currentIndexChanged.connect(self.updateNsources)
        self.source_combo = source_combo

        # self.source_combo = source_combo
        self.nsoures = int(source_combo.currentText())
        logging.info(f"Number of sources: {self.nsoures}")

        self.input_dict = {}
        input_labels = ["RA", "Dec", "Flux Ratio"]
        nsources = self.nsoures
        for ldx, label_text in enumerate(input_labels):
            input_label = QLabel(label_text)
            input_label.setMaximumSize(200, 30)
            input_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            input_label.setContentsMargins(0, 5, 0, 0)
            input_layout.addWidget(input_label, ldx+4, 0)
            for i in range(nsources-1):  # Create three input boxes for each label
                if i == 0 and ldx == 2:
                    continue
                input_box = QLineEdit(self)
                input_box.textChanged.connect(self.inputTextChanged)
                input_box.setMaximumSize(100, 30)
                self.input_dict[f"{label_text} {i+1}"] = ""  # Initialize as empty string
                input_layout.addWidget(input_box, ldx+4, i+1)
        
        for ndx in range(nsources-1):
            star_label = QLabel(f"Source {ndx+2}")
            star_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            star_label.setMaximumSize(100, 30)
            input_layout.addWidget(star_label, 3, ndx+1)
        
        fit_labels = ["Fit Position", "Fit Flux Ratio"]
        for ldx, label_text in enumerate(fit_labels):
            fit_label = QLabel(label_text)
            fit_label.setMaximumSize(200, 30)
            fit_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            fit_label.setContentsMargins(0, 5, 0, 0)
            input_layout.addWidget(fit_label, ldx+7, 0)
            for i in range(nsources-1):
                if i == 0 and ldx == 1:
                    continue
                if nsources == 2 and i == 1:
                    continue
                checkbox = QCheckBox()
                checkbox.setFixedSize(100, 30)
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(self.checkboxStateChanged)
                input_layout.addWidget(checkbox, ldx+7, i+1)
        
        init_labels = ["pc RA", "pc Dec", "alpha central source",
                       "fr Source 1 / Source 2", "flux BG"]
        for ldx, label_text in enumerate(init_labels):
            init_label = QLabel(label_text)
            init_label.setMaximumSize(200, 30)
            init_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            init_label.setContentsMargins(0, 5, 0, 0)
            input_layout.addWidget(init_label, ldx+9, 0)
            init_box = QLineEdit(self)
            init_box.textChanged.connect(self.inputTextChanged)
            init_box.setMaximumSize(100, 30)
            self.input_dict[f"{label_text} {i+1}"] = ""  # Initialize as empty string
            input_layout.addWidget(init_box, ldx+9, 1)

        mode_label = QLabel('Fitting Mode')
        mode_label.setMaximumSize(200, 30)
        mode_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        mode_label.setContentsMargins(0, 5, 0, 0)
        input_layout.addWidget(mode_label, 14, 0)

        mode_combo = QComboBox(self)
        mode_combo.setMaximumSize(150, 30)
        mode_combo.addItems(["Least Sqr", "MCMC"])
        try:
            mode_combo.setCurrentText(str(self.minimizer))
        except AttributeError:
            mode_combo.setCurrentIndex(0)
        self.minimizer = mode_combo.currentText()
        input_layout.addWidget(mode_combo, 14, 1)
        mode_combo.currentIndexChanged.connect(self.updateNsources)
        self.mode_combo = mode_combo

        if self.minimizer == "MCMC":
            mcmc_labels = ["nthreads", "nwalkers", "nsteps"]
            for ldx, label_text in enumerate(mcmc_labels):
                mcmc_label = QLabel(label_text)
                mcmc_label.setMaximumSize(200, 30)
                mcmc_label.setAlignment(Qt.AlignmentFlag.AlignRight)
                mcmc_label.setContentsMargins(0, 5, 0, 0)
                input_layout.addWidget(mcmc_label, ldx+15, 0)
                mcmc_box = QLineEdit(self)
                mcmc_box.textChanged.connect(self.inputTextChanged)
                mcmc_box.setMaximumSize(100, 30)
                self.input_dict[f"{label_text} {i+1}"] = ""  # Initialize as empty string
                input_layout.addWidget(mcmc_box, ldx+15, 1)

        self.input_layout = input_layout
        self.left_layout.addLayout(self.input_layout, 0, 0)

    def updateNsources(self):
        self.nsoures = int(self.source_combo.currentText())
        self.minimizer = self.mode_combo.currentText()
        self.createSourceLayout()
        
    def showFileDialog(self):
        # Create a file dialog and set its properties
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open File")
        # Show the file dialog and get the selected file(s)
        file_paths, _ = file_dialog.getOpenFileNames(self, "Open File", "", "All Files (*);;Text Files (*.txt)")
        if file_paths:
            # Display the selected file(s) in the QLineEdit widget
            self.file_path_edit.setText("\n".join(file_paths))
        try:
            self.data = LoadData(file_paths[0])
        except IndexError:
            pass
        self.updatePlot()

    def checkboxStateChanged(self, state):
        sender = self.sender()
        label_text = sender.text()
        checked = state == 2  # 2 corresponds to checked state in Qt
        self.checkbox_dict[label_text] = checked
        logging.info(f"Checkbox state changed: {label_text} is checked: {checked}")


    def inputTextChanged_old(self, text):
        sender = self.sender()
        label_text = self.input_dict.keys()[list(self.input_dict.values()).index(sender.text())]
        self.input_dict[label_text] = text
        logging.info(f"Input text changed: {label_text} value: {text}")

    def inputTextChanged(self, text):
        sender = self.sender()
        for key, value in self.input_dict.items():
            if self.input_dict[key] != sender.text():
                self.input_dict[key] = sender.text()
                print(key, sender.text())

def main():
    app = QApplication(sys.argv)
    window = GRAVITYfitGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
