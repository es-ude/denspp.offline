import time

import src_gui.setup_gui as gui
import src_gui.usb_serial as usb


if __name__ == "__main__":
    print("GUI for Accessing the SpAIke demonstrator system")
    normal_start = True

    if(normal_start):
        gui.start_gui()
    else:
        dev = usb.com_usb("COM9", 115200)
        dev.setup_usb()
        dev.open()
        for x in range(0, 2):
            print(x)
            dev.write(b'\xAA')
            time.sleep(1)

        dev.close()


