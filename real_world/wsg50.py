from .realur5_utils import connect, Gripper
from time import time, sleep
import atexit


class WSG50(Gripper):
    # http://wsg50-00004407.local/
    BUFFER_SIZE = 1024
    TIMEOUT = 2  # seconds
    VERBOSE = True

    def __init__(self, tcp_ip, tcp_port=1001):
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.tcp_sock = connect(self.tcp_ip, self.tcp_port)
        atexit.register(self.__del__)
        self.ack_fast_stop()
        self.set_clamp_travel()

    def close(self, blocking=True, ** kwargs):
        self.grip(blocking=blocking, ** kwargs)

    def open(self, blocking=True, ** kwargs):
        self.move(30, blocking=blocking, **kwargs)

    def wait_for_msg(self, msg):
        msg = msg.decode("utf-8")
        since = time()
        while True:
            data = self.tcp_sock.recv(self.BUFFER_SIZE)
            data = data.decode("utf-8")
            if msg in data:
                ret = True
                break
            elif data.startswith("ERR"):
                ret = False
                if WSG50.VERBOSE:
                    print(f"[WSG] Error: {data}")
                break
            if time() - since >= self.TIMEOUT:
                if WSG50.VERBOSE:
                    print(f"[WSG] Timeout ({self.TIMEOUT} s) occurred.")
                break
            sleep(0.1)
        return ret

    def ack_fast_stop(self):
        MESSAGE = str.encode("FSACK()\n")
        self.tcp_sock.send(MESSAGE)
        return self.wait_for_msg(b"ACK FSACK\n")

    def set_verbose(self, verbose=True, blocking=True):
        """
        Set verbose True for detailed error messages
        """
        MESSAGE = str.encode(f"VERBOSE={1 if verbose else 0}\n")
        self.tcp_sock.send(MESSAGE)
        if blocking:
            return self.wait_for_msg(MESSAGE)

    def home(self, blocking=True):
        """
        Fully open the gripper
        """
        MESSAGE = str.encode("HOME()\n")
        self.tcp_sock.send(MESSAGE)
        if blocking:
            return self.wait_for_msg(b"FIN HOME\n")

    def move(self, position, speed=200, blocking=True):
        """
        Move fingers to specific position
        * position 0 :- fully close
        * position 110 :- fully open
        """
        MESSAGE = str.encode(f"MOVE({position}, {speed})\n")
        self.tcp_sock.send(MESSAGE)
        if blocking:
            return self.wait_for_msg(b"FIN MOVE\n")

    def set_clamp_travel(self, value=10):
        MESSAGE = str.encode(f"CLT={value}\n")
        self.tcp_sock.send(MESSAGE)

    def grip(self, force=80, part_width=16, blocking=True):
        MESSAGE = str.encode(f"GRIP({force},{part_width})\n")
        self.tcp_sock.send(MESSAGE)
        if blocking:
            return self.wait_for_msg(b"FIN GRIP\n")

    def release(self, part_width=10, speed=200, blocking=True):
        """
        Release: Release object by opening fingers
        """
        MESSAGE = str.encode(f"RELEASE({part_width},{speed})\n")
        self.tcp_sock.send(MESSAGE)
        if blocking:
            return self.wait_for_msg(b"FIN RELEASE\n")

    def bye(self):
        MESSAGE = str.encode("BYE()\n")
        self.tcp_sock.send(MESSAGE)
        return

    def __del__(self):
        self.bye()

    @property
    def ee_tip_z_offset(self):
        # in meters
        return 0.174

    @property
    def current_width(self):
        # TODO
        raise NotImplementedError()
