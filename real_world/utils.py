import cv2
import numpy as np
from environment.utils import get_largest_component
from copy import deepcopy
from real_world.setup import (WORKSPACE_SURFACE, WS_PC)


class InvalidDepthException(Exception):
    def __init__(self):
        super().__init__('Invalid Depth Point')


def bound_grasp_pos(pos, z_offset=0.05):
    pos = deepcopy(pos)
    # grasp slightly lower than detected depth
    pos[2] -= z_offset
    pos[2] = max(WORKSPACE_SURFACE, pos[2])
    pos[2] = min(WORKSPACE_SURFACE+0.1, pos[2])
    return pos


def get_workspace_crop(img):
    retval = img[WS_PC[0]:WS_PC[1], WS_PC[2]:WS_PC[3], ...]
    return retval


def get_cloth_mask(rgb):
    h, w, c = rgb.shape
    if h == 720 and w == 1280:
        rgb[:WS_PC[0], ...] = 0
        rgb[WS_PC[1]:, ...] = 0
        rgb[:, :WS_PC[2], :] = 0
        rgb[:, WS_PC[3]:, :] = 0
    """
    Segments out black backgrounds
    """
    bottom = (0, 0, 0)
    top = (255, 255, 125)
    mask = cv2.inRange(cv2.cvtColor(
        rgb, cv2.COLOR_RGB2HSV), bottom, top)
    mask = (mask == 0).astype(np.uint8)
    if mask.shape[0] != mask.shape[1]:
        mask[:, :int(mask.shape[1]*0.2)] = 0
        mask[:, -int(mask.shape[1]*0.2):] = 0
    return get_largest_component(mask).astype(np.uint8)


def compute_coverage(rgb):
    mask = get_cloth_mask(rgb=rgb)
    return np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])


def pix_to_3d_position(
        x, y, depth_image, cam_intr, cam_extr, cam_depth_scale):
    # Get click point in camera coordinates
    click_z = depth_image[y, x] * cam_depth_scale
    click_x = (x-cam_intr[0, 2]) * \
        click_z/cam_intr[0, 0]
    click_y = (y-cam_intr[1, 2]) * \
        click_z/cam_intr[1, 1]
    if click_z == 0:
        raise InvalidDepthException
    click_point = np.asarray([click_x, click_y, click_z])
    click_point = np.append(click_point, 1.0).reshape(4, 1)

    # Convert camera coordinates to robot coordinates
    target_position = np.dot(cam_extr, click_point)
    target_position = target_position[0:3, 0]
    return target_position


def pick_place_primitive_helper(ur5, pick_pose, place_pose,
                                backup=0.02, **kwargs):
    ur5.gripper.open(blocking=True)
    pick_pose = deepcopy(pick_pose)
    if not ur5.movej(
            params=pick_pose, blocking=True,
            use_pos=True, **kwargs):
        return False
    ur5.gripper.close(blocking=True)
    post_grasp_pose = deepcopy(pick_pose)
    post_grasp_pose[2] += backup
    post_grasp_kwargs = deepcopy(kwargs)
    post_grasp_kwargs['j_vel'] = 0.01
    post_grasp_kwargs['j_acc'] = 0.01
    if not ur5.movel(params=post_grasp_pose, blocking=True, use_pos=True,
                     **post_grasp_kwargs):
        return False
    if not ur5.movej(
            params=place_pose, blocking=True,
            use_pos=True, **kwargs):
        return False
    ur5.gripper.open(blocking=True)
    return True
