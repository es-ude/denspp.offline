import sys
import PyQt5.QtWidgets
import PyQt5.QtCore
import PyQt5.QtGui

class GUI(PyQt5.QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(GUI, self).__init__(parent)
        # --- Values for the frame
        self.__width = 1024
        self.__height = 480
        self.__textSize = 20
        # --- Properties of text style
        font = PyQt5.QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(self.__textSize)
        # --- Frame properties
        self.setGeometry(10, 40, self.__width, self.__height)
        self.setWindowTitle("Beispiel: Oberfläche mit PyQt5 in Python")

        # --- Variables for application
        self.useAutoEncoder = False

        # Element: Text
        label = PyQt5.QtWidgets.QLabel(self)
        label.setText("Hallo Welt!")
        label.setFont(font)
        label.move(50, 20)
        # Element: Button
        self.b0 = PyQt5.QtWidgets.QPushButton(self)
        self.b0.setText("Test")
        self.b0.move(50, 50)
        self.b0.clicked.connect(self.__runButton0)
        # Element: Button
        self.b1 = PyQt5.QtWidgets.QPushButton(self)
        self.b1.setText("Beenden")
        self.b1.move(900, 400)
        self.b1.clicked.connect(self.__runButton1)
        # Element: CheckBox
        self.c0 = PyQt5.QtWidgets.QCheckBox(self)
        self.c0.setText("Beenden")
        self.c0.move(400, 400)
        self.c0.clicked.connect(self.__runCheckBox0)

    def __runButton0(self) -> None:
        print("Test - Button 0")

    def __runButton1(self) -> None:
        self.close()

    def __runCheckBox0(self) -> None:
        self.useAutoEncoder = self.c0.isChecked()
        print(f"Use AutoEncoder: {self.useAutoEncoder}")

def setup():
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    ex = GUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    setup()