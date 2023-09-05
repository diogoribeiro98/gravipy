import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox, QFileDialog, QLineEdit, QComboBox, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle("Data Plotter")
        self.setGeometry(100, 100, 800, 500)

        # Create a button to open the file dialog
        self.button = QPushButton("Open File Dialog", self)
        self.button.setGeometry(20, 20, 150, 40)
        self.button.clicked.connect(self.showFileDialog)

        # Create a QLineEdit widget to display the selected file path(s)
        self.file_path_edit = QLineEdit(self)
        self.file_path_edit.setGeometry(200, 20, 400, 30)
        self.file_path_edit.setPlaceholderText("Selected File(s)")

        # Create a combo box (dropdown) to select the data source
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.setGeometry(20, 80, 200, 30)
        self.data_source_combo.addItems(["Data Source 1", "Data Source 2", "Data Source 3", "Data Source 4"])
        self.data_source_combo.currentIndexChanged.connect(self.updatePlot)

        # Create a Matplotlib canvas for plotting
        self.canvas = PlotCanvas(self)
        self.canvas.setGeometry(20, 120, 760, 340)

    def showFileDialog(self):
        # Create a file dialog and set its properties
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open File")

        # Show the file dialog and get the selected file(s)
        file_paths, _ = file_dialog.getOpenFileNames(self, "Open File", "", "All Files (*);;Text Files (*.txt)")

        if file_paths:
            # Display the selected file(s) in the QLineEdit widget
            self.file_path_edit.setText("\n".join(file_paths))

    def updatePlot(self):
        # Dummy data generation, replace with "gravipy" data loading
        selected_data_source = self.data_source_combo.currentText()
        if selected_data_source == "Data Source 1":
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
        elif selected_data_source == "Data Source 2":
            x = np.linspace(0, 10, 100)
            y = np.cos(x)
        elif selected_data_source == "Data Source 3":
            x = np.linspace(0, 10, 100)
            y = x ** 2
        elif selected_data_source == "Data Source 4":
            x = np.linspace(0, 10, 100)
            y = np.exp(x)

        self.canvas.plotData(x, y)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)

    def plotData(self, x, y):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title("Data Plot")
        self.draw()

def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
