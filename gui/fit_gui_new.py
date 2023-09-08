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
except ImportError:
    from PyQt5.QtWidgets import (QApplication, QMainWindow,
                                QPushButton, QMessageBox,
                                QFileDialog, QLineEdit,
                                QComboBox, QLabel,
                                QTextEdit, QFormLayout,
                                QVBoxLayout, QHBoxLayout,
                                QWidget, QCheckBox,
                                )
    from PyQt5.QtCore import Qt
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

        self.fitlabel = QLabel("Fitting options:", self)

        # Create input boxes and a dictionary to store their values
        self.input_dict = {}
        input_labels = ["RA", "Dec", "Flux Ratio"]
        input_layout = QVBoxLayout()  # To hold RA, Dec, Flux Ratio

        for label_text in input_labels:
            input_box = QLineEdit(self)
            input_box.textChanged.connect(self.inputTextChanged)
            self.input_dict[label_text] = ""  # Initialize as empty string
            input_layout.addWidget(QLabel(label_text))
            input_layout.addWidget(input_box)

        # Create checkboxes with labels and a dictionary to store their states
        self.checkbox_dict = {}
        checkbox_labels = ["Checkbox 1", "Checkbox 2", "Checkbox 3"]
        checkbox_layout = QVBoxLayout()
        for label_text in checkbox_labels:
            checkbox = QCheckBox(label_text, self)
            checkbox.stateChanged.connect(self.checkboxStateChanged)
            self.checkbox_dict[label_text] = False  # Initialize as unchecked
            checkbox_layout.addWidget(checkbox)

        main_layout = QGridLayout()

        # Create a widget to hold everything in the right half
        right_widget = QWidget(self)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.plotlabel)
        right_layout.addWidget(self.data_source_combo)
        right_layout.addWidget(self.canvas1)
        right_layout.addWidget(self.canvas2)
        right_layout.addWidget(self.log_text_edit)

        left_widget = QWidget(self)
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_layout.addWidget(self.fitlabel)
        left_layout.addLayout(input_layout)
        left_layout.addLayout(checkbox_layout)

        main_layout.addWidget(self.button, 0, 0, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self.file_path_edit, 0, 1)
        main_layout.addLayout(right_layout, 1, 1)
        main_layout.addLayout(left_layout, 1, 0)

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

    def inputTextChanged(self, text):
        sender = self.sender()
        label_text = self.input_dict.keys()[list(self.input_dict.values()).index(sender.text())]
        self.input_dict[label_text] = text
        logging.info(f"Input text changed: {label_text} value: {text}")

def main():
    app = QApplication(sys.argv)
    window = GRAVITYfitGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
