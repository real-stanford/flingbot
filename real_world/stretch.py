import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2

GRIPPER_LINE = 280
CLOTH_LINE = 420
FOREGROUND_BACKGROUND_DIST = 1.0


def is_cloth_grasped(depth):
    cloth_mask = cv2.morphologyEx(
        np.logical_and(
            depth < 1.2, depth != 0).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=4)

    gripper_strip = cloth_mask[GRIPPER_LINE, :]
    # find grippers
    center = len(gripper_strip)//2
    right_gripper_pix = center + 1
    while not gripper_strip[right_gripper_pix]:
        right_gripper_pix += 1
        if right_gripper_pix == len(gripper_strip) - 1:
            break
    left_gripper_pix = center - 1
    while not gripper_strip[left_gripper_pix]:
        left_gripper_pix -= 1
        if left_gripper_pix == 0:
            break
    center = int((left_gripper_pix + right_gripper_pix)/2)
    cloth_mask[:, :max(left_gripper_pix-100, 1)] = 0
    cloth_mask[:, min(right_gripper_pix+100, cloth_mask.shape[1]):] = 0
    left_grasped = cloth_mask[CLOTH_LINE, :center].sum() > 0
    right_grasped = cloth_mask[CLOTH_LINE, center:].sum() > 0
    return [left_grasped, right_grasped]


def plt_batch(imgs, title=''):
    fig, axes = plt.subplots(3, 3)
    fig.set_figheight(6)
    fig.set_figwidth(7)
    fig.suptitle(title)
    for ax, (img, title) in zip(axes.flatten(), imgs):
        ax.set_title(title)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def is_cloth_stretched(
        rgb, depth,
        angle_tolerance=20, threshold=20, debug=False):
    from environment.utils import get_largest_component
    imshows = []
    fgbg = np.logical_and(
        depth < FOREGROUND_BACKGROUND_DIST, depth != 0).astype(np.uint8)
    if debug:
        imshows = [(rgb, 'rgb'), (depth, 'depth'), (fgbg.copy(), 'fgbg')]
    fgbg = cv2.morphologyEx(
        fgbg, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=4)
    if debug:
        imshows.append((fgbg.copy(), 'mask'))
    gripper_strip = fgbg[GRIPPER_LINE, :]
    # find grippers
    center = len(gripper_strip)//2
    right_gripper_pix = center + 1
    while not gripper_strip[right_gripper_pix]:
        right_gripper_pix += 1
        if right_gripper_pix == len(gripper_strip) - 1:
            break
    left_gripper_pix = center - 1
    while not gripper_strip[left_gripper_pix]:
        left_gripper_pix -= 1
        if left_gripper_pix == 0:
            break
    center = int((left_gripper_pix + right_gripper_pix)/2)
    fgbg[:, :left_gripper_pix] = 0
    fgbg[:, right_gripper_pix:] = 0
    fgbg[:GRIPPER_LINE, :] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5))
    line_mask = cv2.morphologyEx(
        fgbg.copy(), cv2.MORPH_CLOSE, kernel,
        iterations=4)

    kernel = np.array([[-1], [0], [1]]*3)
    line_mask = cv2.filter2D(fgbg, -1, kernel)
    if debug:
        imshows.append((fgbg.copy(), 'filtered'))
        imshows.append((line_mask.copy(), 'horizontal edges'))
    line_mask = get_largest_component(
        cv2.morphologyEx(
            line_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (10, 10)), iterations=5))
    if debug:
        imshows.append((line_mask.copy(), 'largest component'))
    # find angle to rotate to
    points = np.array(np.where(line_mask.copy() == 1)).T
    points = np.array(sorted(points, key=lambda x: x[1]))
    max_x = points[-1][1]
    min_x = points[0][1]
    min_x_y = min(points[(points[:, 1] == min_x)],
                  key=lambda pnt: pnt[0])[0]
    max_x_y = min(points[(points[:, 1] == max_x)],
                  key=lambda pnt: pnt[0])[0]
    angle = 180 * np.arctan((max_x_y-min_x_y)/(max_x - min_x))/np.pi
    line_mask = ndimage.rotate(line_mask, angle, reshape=False)
    if debug:
        print('angle:', angle)
        img = np.zeros(line_mask.shape).astype(np.uint8)
        img = cv2.circle(
            img=img,
            center=(min_x, min_x_y),
            radius=10, color=1, thickness=3)
        img = cv2.circle(
            img=img,
            center=(max_x, max_x_y),
            radius=10, color=1, thickness=3)
        imshows.append((img, 'circled'))
        imshows.append((line_mask.copy(), f'rotated ({angle:.02f}°)'))
    # if angle is too sharp, cloth is probably not stretched
    y_values = np.array(np.where(line_mask == 1))[0, :]
    min_coord = y_values.min()
    max_coord = y_values.max()
    stretchedness = 1/((max_coord - min_coord)/line_mask.shape[0])
    too_tilted = np.abs(angle) > angle_tolerance
    stretch = (not too_tilted) and (stretchedness > threshold)
    if debug:
        print(stretchedness)
        plt_batch(
            imshows, f'Stretchness: {stretchedness:.02f}, Stretched: {stretch}')
    return stretch


def stretch(ur5_pair, front_camera, height: float, grasp_width: float,
            max_grasp_width=0.6):
    from .setup import DEFAULT_ORN, DIST_UR5
    while True:
        rgb, depth = front_camera.get_rgbd(repeats=3)
        # if both arms no longer are holding onto cloth anymore
        if not all(is_cloth_grasped(depth=depth)) \
                or is_cloth_stretched(rgb=rgb, depth=depth) or \
                grasp_width > max_grasp_width:
            return grasp_width
        grasp_width += 0.02
        dx = (DIST_UR5 - grasp_width)/2
        ur5_pair.movel(
            params=[
                # left
                [dx, 0, height] + DEFAULT_ORN,
                # right
                [dx, 0, height] + DEFAULT_ORN],
            blocking=True,
            use_pos=True)
