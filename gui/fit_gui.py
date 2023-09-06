import sys
import numpy as np
# from PyQt6.QtWidgets import (QApplication, QMainWindow,
#                              QPushButton, QMessageBox,
#                              QFileDialog, QLineEdit)

from PyQt6.QtWidgets import (QApplication, QMainWindow,
                             QPushButton, QMessageBox,
                             QFileDialog, QLineEdit,
                             QComboBox, QLabel,
                             QTextEdit,
                             QVBoxLayout, 
                             QWidget
                            )

from matplotlib.figure import Figure
from gui_utils import PlotData, LoggingHandler, LoadData
import logging


class GRAVITYfitGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle("GRAVITY multi source fitting")
        # q: what are the parameters of setGeometry?
        # a: x, y, width, height
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
        self.label = QLabel("Plot Quantity:", self)
        self.label.setGeometry(20, 80, 150, 30)
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.setGeometry(20, 110, 150, 30)
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
        logging.info(f"Window size: {window_width} x {window_height}")

        self.file_path_edit.setGeometry(200, 20, 
                                        window_width-220, 30)
        self.log_text_edit.setGeometry(20, window_height-180, 
                                       window_width//2-40, 150)

        can_width = (window_width//2-40)
        can_height = (window_height-350) // 2
        self.canvas1.setGeometry(20, 150,
                                 can_width, can_height)
        self.canvas2.setGeometry(20, 150+can_height,
                                 can_width, can_height)

        # # Adjust the file path edit position and size
        # edit_x = window_width * 0.4
        # edit_y = window_height * 0.05
        # edit_width = window_width * 0.5
        # edit_height = window_height * 0.05
        # self.file_path_edit.setGeometry(edit_x, edit_y, edit_width, edit_height)

        # # Adjust the combo box position and size
        # combo_x = window_width * 0.1
        # combo_y = window_height * 0.2
        # combo_width = window_width * 0.2
        # combo_height = window_height * 0.05
        # self.data_source_combo.setGeometry(combo_x, combo_y, combo_width, combo_height)

        # # Adjust the canvas position and size
        # canvas_x = window_width * 0.05
        # canvas_y = window_height * 0.3
        # canvas_width = window_width * 0.9
        # canvas_height = window_height * 0.6
        # self.canvas.setGeometry(canvas_x, canvas_y, canvas_width, canvas_height)

        # # Redraw the plot
        # self.updatePlot()

def main():
    app = QApplication(sys.argv)
    window = GRAVITYfitGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
