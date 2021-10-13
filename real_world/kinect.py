import requests
import pickle

# See https://github.com/columbia-ai-robotics/PyKinect


class KinectClient:
    def __init__(self, ip='XXX.XXX.X.XXX', port=8080):
        self.ip = ip
        self.port = port

    @property
    def color_intr(self):
        return self.get_intr()

    def get_intr(self):
        return pickle.loads(requests.get(f'http://{self.ip}:{self.port}/intr').content)

    def get_rgbd(self, repeats=10):
        data = pickle.loads(requests.get(
            f'http://{self.ip}:{self.port}/pickle/{repeats}').content)
        return data['color_img'], data['depth_img']
