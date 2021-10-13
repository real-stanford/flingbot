from real_world.kinect import KinectClient
from real_world.realsense import RealSense
from real_world.realur5 import UR5
from real_world.wsg50 import WSG50
from real_world.rg2 import RG2

DEFAULT_ORN = [2.22, 2.22, 0.0]
DIST_UR5 = 1.34
WORKSPACE_SURFACE = -0.15
MIN_GRASP_WIDTH = 0.25
MAX_GRASP_WIDTH = 0.6
MIN_UR5_BASE_SAFETY_RADIUS = 0.3
# workspace pixel crop
WS_PC = [30, -165, 385, -370]

UR5_VELOCITY = 0.5
UR5_ACCELERATION = 0.3

CLOTHS_DATASET = {
    'hannes_tshirt': {
        'flatten_area': 0.0524761,
        'cloth_size': (0.45, 0.55),
        'mass': 0.2
    },
}
CURRENT_CLOTH = 'hannes_tshirt'


def get_ur5s():
    return [
        UR5(tcp_ip='XXX.XXX.X.XXX',
            velocity=UR5_VELOCITY,
            acceleration=UR5_ACCELERATION,
            gripper=RG2(tcp_ip='XXX.XXX.X.XXX')),
        UR5(tcp_ip='XXX.XXX.X.XXX',
            velocity=UR5_VELOCITY,
            acceleration=UR5_ACCELERATION,
            gripper=WSG50(tcp_ip='XXX.XXX.X.XXX')),
    ]


def get_top_cam():
    return KinectClient()


def get_front_cam():
    return RealSense(
        tcp_ip='127.0.0.1',
        tcp_port=12345,
        im_h=720,
        im_w=1280,
        max_depth=3.0)
