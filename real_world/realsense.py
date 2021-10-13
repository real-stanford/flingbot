import socket
import numpy as np
import threading
import time


class RealSense(object):

    def __init__(self, tcp_ip, tcp_port, im_h, im_w, max_depth):
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port
        self.im_h = im_h
        self.im_w = im_w
        self.max_depth = max_depth  # in meters
        self.buffer_size = 10*4 + self.im_h*self.im_w*5  # in bytes

        # Connect to server
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((self.tcp_ip, self.tcp_port))

        # Fetch data continually
        self.color_im = None
        self.depth_im = None
        self.timestamp = None
        self.color_intr = None
        self.depth_intr = None
        self.depth2color_extr = None
        capture_thread = threading.Thread(target=self.get_data)
        capture_thread.daemon = True
        capture_thread.start()

        while self.color_im is None or self.depth_im is None:
            time.sleep(0.01)

    def get_data(self):
        while True:
            # Ping the server with anything
            self.tcp_sock.send(b'42')

            data = b''
            while len(data) < (9*4+9*4+16*4+4+8+self.im_h*self.im_w*5):
                data += self.tcp_sock.recv(self.buffer_size)

            # Re-organize TCP data into color and depth frame
            self.color_intr = np.frombuffer(
                data[0:(9*4)], np.float32).reshape(3, 3)
            self.depth_intr = np.frombuffer(
                data[(9*4):(9*4+9*4)], np.float32).reshape(3, 3)
            self.depth2color_extr = np.frombuffer(
                data[(9*4+9*4):(9*4+9*4+16*4)], np.float32).reshape(4, 4)
            self.depth_scale = np.frombuffer(
                data[(9*4+9*4+16*4):(9*4+9*4+16*4+4)], np.float32)[0]
            self.timestamp = np.frombuffer(
                data[(9*4+9*4+16*4+4):(9*4+9*4+16*4+4+8)], np.long)[0]
            depth_im = np.frombuffer(data[(9*4+9*4+16*4+4+8):(
                (9*4+9*4+16*4+4+8)+self.im_w*self.im_h*2)],
                np.uint16).reshape(self.im_h, self.im_w)
            self.color_im = np.frombuffer(
                data[((9*4+9*4+16*4+4+8)+self.im_w*self.im_h*2):],
                np.uint8).reshape(self.im_h, self.im_w, 3)
            depth_im = depth_im.copy().astype(float)   # * depth_scale
            depth_im /= 10000

            # Set invalid depth pixels to zero
            depth_im[depth_im > self.max_depth] = 0.0
            self.depth_im = depth_im

    def get_rgbd(self, repeats=10):
        rgbs = []
        depths = []
        for _ in range(repeats):
            rgbs.append(self.color_im.copy())
            depths.append(self.depth_im.copy())
            time.sleep(0.05)
        rgb = np.mean(rgbs, axis=0).astype(np.uint8)
        depth = np.zeros(depths[0].shape)
        count = np.zeros(depths[0].shape)
        for img in depths:
            depth[img != 0] += img[img != 0]
            count[img != 0] += 1
        # for pixels with few points, just set to zero
        depth[count < 0.5*repeats] = 0
        count[depth == 0] = 1
        depth = depth/count
        return rgb, depth
