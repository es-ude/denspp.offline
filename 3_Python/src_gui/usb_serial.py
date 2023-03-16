import serial

class com_usb():
    def __init__(self, com_name: str, baud: int):
        self.SerialName = com_name
        self.SerialBaud = baud

        self.device = None
        self.device_init = False

    def setup_usb(self):
        self.device = serial.Serial(
            port=self.SerialName,
            baudrate=self.SerialBaud,
            parity=serial.PARITY_ODD,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )

    def open(self):
        if(self.device.is_open):
            self.device.close()
        self.device.open()

    def close(self):
        self.device.close()

    def write(self, input):
        self.device.write(serial.to_bytes(input))



