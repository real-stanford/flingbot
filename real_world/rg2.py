from .realur5_utils import connect, Gripper, skip_to_package_index
from time import sleep, time
import struct
import socket


class RG2(Gripper):
    def __init__(self, tcp_ip, tcp_port=30002):
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.tcp_sock = connect(self.tcp_ip, self.tcp_port)

    def open(self, blocking=True, timeout=3.0, width=0.3):
        tcp_msg = "set_digital_out(8,False)\n"
        self.tcp_sock.send(str.encode(tcp_msg))
        start = time()
        if blocking:
            sleep(0.5)
            return True
        while blocking and self.current_width < width:
            sleep(0.05)
            if float(time()-start) > timeout:
                return False
        return True

    def close(self, blocking=True, timeout=3.0, width=0.1):
        tcp_msg = "set_digital_out(8,True)\n"
        self.tcp_sock.send(str.encode(tcp_msg))
        start = time()
        if blocking:
            sleep(0.75)
            return True
        while blocking and self.current_width > width:
            sleep(0.05)
            if float(time()-start) > timeout:
                return False
        return True

    @property
    def ee_tip_z_offset(self):
        return 0.213

    @property
    def current_width(self):
        # TODO https://onrobot.com/sites/default/files/documents/RG2_User%20_Manual_enEN_V1.9.2.pdf
        state_data = self.get_state_data()
        byte_index = skip_to_package_index(state_data, pkg_type=3)+14
        analog_input1 = struct.unpack(
            '!d', state_data[(byte_index+0):(byte_index+8)])[0]
        # Find peak in analog input
        timeout_t0 = time()
        while True:
            state_data = self.get_state_data()
            byte_index = skip_to_package_index(
                state_data, pkg_type=3)+14
            new_analog_input1 = struct.unpack(
                '!d', state_data[(byte_index+0):(byte_index+8)])[0]
            timeout_t1 = time()
            if (new_analog_input1 > 2.0 and
                abs(new_analog_input1 - analog_input1) > 0.0 and
                abs(new_analog_input1 - analog_input1) < 0.1)\
                    or timeout_t1 - timeout_t0 > 5:
                return analog_input1
            analog_input1 = new_analog_input1
            sleep(0.1)

    def get_state_data(self):
        max_tcp_msg_size = 2048
        since = time()
        while True:
            if time() - since < 3:
                message_size_bytes = bytearray(self.tcp_sock.recv(4))
                message_size = struct.unpack("!i", message_size_bytes)[0]
                # This is hacky but it can work for multiple versions
                if message_size <= 55 or message_size >= max_tcp_msg_size:
                    continue
                else:
                    state_data = self.tcp_sock.recv(message_size-4)
                if message_size < max_tcp_msg_size and\
                        message_size-4 == len(state_data):
                    return state_data
            else:
                print('Timeout: retrieving TCP message exceeded 3 seconds.',
                      ' Restarting connection.')
                self.__rtc_sock = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.__rtc_sock.connect((self.__tcp_ip, self.__rtc_port))
                break
