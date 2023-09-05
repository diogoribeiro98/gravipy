import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle("PyQt6 Example")
        self.setGeometry(100, 100, 400, 200)

        # Create a button and connect it to a function
        self.button = QPushButton("Click Me!", self)
        self.button.setGeometry(150, 80, 100, 40)
        self.button.clicked.connect(self.showMessageBox)

    def showMessageBox(self):
        # Display a message box when the button is clicked
        msgBox = QMessageBox()
        msgBox.setText("Button Clicked!")
        msgBox.exec()

def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()