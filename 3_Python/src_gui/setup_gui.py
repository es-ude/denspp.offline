import sys, time
import PyQt5.QtWidgets
import PyQt5.QtCore
import PyQt5.QtGui

import src_gui.usb_serial as usb

dev = usb.com_usb("COM9", 115200)

class GUI(PyQt5.QtWidgets.QWidget):
    def __init__(self):
        super(GUI, self).__init__()
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
        self.usb_device = None
        self.useAutoEncoder = False

        # Element: Text
        label = PyQt5.QtWidgets.QLabel(self)
        label.setText("GUI für Demonstrator-Setup Sp:AI:ke")
        label.setFont(font)
        label.move(50, 20)
        # Element: Button
        self.b0 = PyQt5.QtWidgets.QPushButton(self)
        self.b0.setText("Starte Test")
        self.b0.move(40, 100)
        self.b0.clicked.connect(self.__runButton0)
        # Element: Button
        self.b0 = PyQt5.QtWidgets.QPushButton(self)
        self.b0.setText("Init. Device")
        self.b0.move(40, 60)
        self.b0.clicked.connect(self.__runInit)
        # Element: CheckBox
        self.c0 = PyQt5.QtWidgets.QCheckBox(self)
        self.c0.setText("Aktiviere Autoencoder")
        self.c0.move(40, 150)
        self.c0.clicked.connect(self.__runCheckBox0)
        # Element: Button
        self.b1 = PyQt5.QtWidgets.QPushButton(self)
        self.b1.setText("Beenden")
        self.b1.move(900, 400)
        self.b1.clicked.connect(self.__runExit)

    def __runInit(self):
        print("Setup Device")
        init_usb()

    def __runButton0(self) -> None:
        print("Test - Button 0")
        write_cmd(b'\xAA')

    def __runCheckBox0(self) -> None:
        self.useAutoEncoder = self.c0.isChecked()
        print(f"Use AutoEncoder: {self.useAutoEncoder}")

    def __runExit(self) -> None:
        dev.close()
        self.close()

def init_usb():
    dev.setup_usb()
    dev.open()

def write_cmd(input):
    dev.write(input)

def close_usb():
    dev.close()

def start_gui():
    app = PyQt5.QtWidgets.QApplication(sys.argv)

    ex = GUI()
    ex.show()
    sys.exit(app.exec_())