from .utils import (
    get_cloth_mask,
    pix_to_3d_position,
    pick_place_primitive_helper,
    InvalidDepthException,
    bound_grasp_pos
)
from .setup import (
    DEFAULT_ORN
)
import numpy as np
import random


def pick_and_drop(ur5_pair, top_camera, top_cam_right_ur5_pose,
                  top_cam_left_ur5_pose, cam_depth_scale):
    before_mask = get_cloth_mask(rgb=top_camera.get_rgbd()[0])
    rgb, depth = top_camera.get_rgbd()
    cloth_mask = get_cloth_mask(rgb=rgb)
    points = np.array(np.where(cloth_mask == 1))

    # Find random point on cloth
    indices = list(range(points.shape[1]))
    random.shuffle(indices)
    for i in indices:
        point = points[:, i]
        y, x = point

        # Try with right arm
        try:
            pick_pos = list(pix_to_3d_position(
                x=x, y=y, depth_image=depth,
                cam_intr=top_camera.color_intr,
                cam_extr=top_cam_right_ur5_pose,
                cam_depth_scale=cam_depth_scale))
            pick_pos = bound_grasp_pos(pick_pos)

            if ur5_pair.right_ur5.check_pose_reachable(
                pose=list(pick_pos) + list(DEFAULT_ORN))\
                and pick_place_primitive_helper(
                    ur5=ur5_pair.right_ur5,
                    pick_pose=list(pick_pos) + list(DEFAULT_ORN),
                    place_pose=[0.65, 0.1, 0.35] + DEFAULT_ORN):
                ur5_pair.out_of_the_way()
                after_mask = get_cloth_mask(rgb=top_camera.get_rgbd()[0])
                intersection = np.logical_and(before_mask, after_mask).sum()
                union = np.logical_or(before_mask, after_mask).sum()
                iou = intersection/union
                if iou < 1 - 2e-1:
                    return
                else:
                    continue
        except InvalidDepthException as e:
            pass

        # Try with left arm
        try:
            pick_pos = list(pix_to_3d_position(
                x=x, y=y, depth_image=depth,
                cam_intr=top_camera.color_intr,
                cam_extr=top_cam_left_ur5_pose,
                cam_depth_scale=cam_depth_scale))
            pick_pos = bound_grasp_pos(pick_pos)

            if ur5_pair.left_ur5.check_pose_reachable(
                pose=list(pick_pos) + list(DEFAULT_ORN))\
                and pick_place_primitive_helper(
                    ur5=ur5_pair.left_ur5,
                    pick_pose=list(pick_pos) + list(DEFAULT_ORN),
                    place_pose=[0.65, 0.1, 0.35] + DEFAULT_ORN):
                ur5_pair.out_of_the_way()
                after_mask = get_cloth_mask(rgb=top_camera.get_rgbd()[0])
                intersection = np.logical_and(before_mask, after_mask).sum()
                union = np.logical_or(before_mask, after_mask).sum()
                iou = intersection/union
                if iou < 1 - 2e-1:
                    return
                else:
                    continue
        except InvalidDepthException:
            pass

    ur5_pair.out_of_the_way()
