import sys
import numpy as np
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow,
                                QPushButton, QMessageBox,
                                QFileDialog, QLineEdit,
                                QComboBox, QLabel,
                                QTextEdit, QFormLayout,
                                QVBoxLayout, QHBoxLayout,
                                QWidget, QCheckBox,
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
        self.button.setGeometry(20, 20, 150, 30)
        self.button.clicked.connect(self.showFileDialog)

        # Create a QLineEdit widget to display the selected file path
        self.file_path_edit = QLineEdit(self)
        self.file_path_edit.setGeometry(0, 0, 0, 0)
        self.file_path_edit.setPlaceholderText("Selected File")

        # Create a combo box (dropdown) to select the data source
        self.plotlabel = QLabel("Plot Quantity:", self)
        self.plotlabel.setGeometry(0, 0, 0, 0)
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.setGeometry(0, 0, 0, 0)
        self.data_source_combo.addItems(["Vis Amp", "Vis 2", "Closure", "Vis Phi"])
        self.data_source_combo.currentIndexChanged.connect(self.updatePlot)
        
        self.canvas1 = PlotData(self)
        self.canvas1.setGeometry(0, 0, 0, 0)
        self.canvas2 = PlotData(self)
        self.canvas2.setGeometry(0, 0, 0, 0)

        # Create a QTextEdit widget to display log messages
        self.log_text_edit = QTextEdit(self)
        self.log_text_edit.setGeometry(0, 0, 0, 0)
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setPlaceholderText("Log Messages")
        self.log_handler = LoggingHandler(self.log_text_edit)
        self.log_handler.setFormatter(logging.Formatter('%(levelname)s: %(name)s - %(message)s'))
        self.log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

        self.fitlabel = QLabel("Fitting options:", self)
        self.fitlabel.setGeometry(20, 80, 150, 30)

        # # Create input boxes and a dictionary to store their values
        # self.input_dict = {}
        # input_labels = ["RA", "Dec", "Flux Ratio"]
        # for i, label_text in enumerate(input_labels):
        #     input_box = QLineEdit(self)
        #     input_box.setGeometry(20, 100 + i * 30, 100, 30)
        #     input_box.textChanged.connect(self.inputTextChanged)
        #     self.input_dict[label_text] = "0.0"  # Initialize as empty string

        # # Create checkboxes with labels and a dictionary to store their states
        # self.checkbox_dict = {}
        # checkbox_labels = ["Checkbox 1", "Checkbox 2", "Checkbox 3"]
        # for label_text in checkbox_labels:
        #     checkbox_layout = QHBoxLayout()
        #     checkbox = QCheckBox(label_text, self)
        #     checkbox.stateChanged.connect(self.checkboxStateChanged)
        #     checkbox_layout.addWidget(checkbox)
        #     self.checkbox_dict[label_text] = False  # Initialize as unchecked
        #     self.addLayout(checkbox_layout)

        # # Create input boxes and a dictionary to store their values
        # self.input_dict = {}
        # input_labels = ["Input 1", "Input 2", "Input 3"]
        # for label_text in input_labels:
        #     input_layout = QFormLayout()
        #     input_box = QLineEdit(self)
        #     input_box.textChanged.connect(self.inputTextChanged)
        #     input_layout.addRow(label_text, input_box)
        #     self.input_dict[label_text] = ""  # Initialize as empty string
        #     self.addLayout(input_layout)

        # resize in cae of window resize
        self.resizeEvent = self.adjustWidgetSizes

    def updatePlot(self):
        selected_data_source = self.data_source_combo.currentText()
        try:
            logging.info(f"Selected data source: {selected_data_source}")
            if self.data.polmode == 'SPLIT':
                self.canvas1.plot_data(selected_data_source, self.data.data,
                                    lowest_plot=False, pol_idx=0)
                self.canvas2.plot_data(selected_data_source, self.data.data,
                                    lowest_plot=True, pol_idx=1)
            else:
                self.canvas1.plot_data(selected_data_source, self.data.data,
                                    lowest_plot=True, pol_idx=0)
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

    def adjustWidgetSizes(self, event):
        # Calculate widget sizes and positions based on the window size
        window_width = self.width()
        window_height = self.height()

        logging.debug(f"Window size: {window_width} x {window_height}")


        self.file_path_edit.setGeometry(200, 20, 
                                        window_width-220, 30)
        self.log_text_edit.setGeometry(window_width//2 + 20, window_height-180, 
                                       window_width//2-40, 150)

        self.plotlabel.setGeometry(window_width//2 + 20, 80, 150, 30)
        self.data_source_combo.setGeometry(window_width//2 + 20, 110, 150, 30)
        
        can_width = (window_width//2-40)
        can_height = (window_height-350) // 2
        self.canvas1.setGeometry(window_width//2 + 20, 150,
                                 can_width, can_height)
        self.canvas2.setGeometry(window_width//2 + 20, 150+can_height,
                                 can_width, can_height)
        
    def checkboxStateChanged(self, state):
        sender = self.sender()
        label_text = sender.text()
        checked = state == 2  # 2 corresponds to checked state in Qt
        self.checkbox_dict[label_text] = checked
        logging.info(f"Checkbox state changed: {label_text} is checked: {checked}")

    def inputTextChanged(self, text):
        sender = self.sender().parentWidget()  # Get the parent layout
        label_text = sender.itemAt(0).widget().text()  # Get the label text
        self.input_dict[label_text] = text
        logging.info(f"Input text changed: {label_text} value: {text}")

def main():
    app = QApplication(sys.argv)
    window = GRAVITYfitGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
