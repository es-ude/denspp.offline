import sys
import PyQt5.QtWidgets
import PyQt5.QtCore
import PyQt5.QtGui

class GUI(PyQt5.QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(GUI, self).__init__(parent)
        # --- Values for the frame
        self.width = 1024
        self.height = 480
        self.textSize = 16
        # --- Variables
        self.useAutoEncoder = False
        # Frame properties
        self.setGeometry(10, 40, self.width, self.height)
        #self.resize(self.width, self.height)
        self.setWindowTitle("Das ist ein Test")
        # Properties of art
        font = PyQt5.QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(self.textSize)

        # Element: Text
        self.label = PyQt5.QtWidgets.QLabel(self)
        self.label.setText("Hallo Welt!")
        self.label.setFont(font)
        self.label.move(50, 20)

        #Element: Button
        b0 = PyQt5.QtWidgets.QPushButton(self)
        b0.setText("Test")
        b0.move(50, 50)
        b0.clicked.connect(self.runButton0)

        # Element: Button
        b1 = PyQt5.QtWidgets.QPushButton(self)
        b1.setText("Beenden")
        b1.move(900, 400)
        b1.clicked.connect(self.runButton1)

        # Element: CheckBox
        self.c0 = PyQt5.QtWidgets.QCheckBox(self)
        self.c0.setText("Beenden")
        self.c0.move(400, 400)
        self.c0.clicked.connect(self.runCheckBox0)

    def runButton0(self) -> None:
        print("Test - Button 0")

    def runButton1(self) -> None:
        self.close()

    def runCheckBox0(self) -> None:
        self.useAutoEncoder = self.c0.isChecked()
        print(f"Use AutoEncoder: {self.useAutoEncoder}")

def setup():
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    ex = GUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    setup()