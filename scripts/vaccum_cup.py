import serial
import time

class VaccumCup():
    def __init__(self, port="/dev/ttyUSB0"):
        try:
            self.s = serial.Serial(port, baudrate=115200)
            self.release()
            time.sleep(3)
        except Exception as e:
            print(e)
        print("serial port state", self.s.isOpen())

    # def release(self):
    #     self.s.write("OFF".encode('utf-8'))
    #     #print(self.s.write(b"OFF"))
    #
    # def grasp(self):
    #     self.s.write("ON".encode('utf-8'))
    #     #print(self.s.write(b"ON"))

    def release(self):
        self.s.write("ON".encode('utf-8'))
        #print(self.s.write(b"OFF"))

    def grasp(self):
        self.s.write("OFF".encode('utf-8'))
        #print(self.s.write(b"ON"))

