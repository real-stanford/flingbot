from torch import cat, tensor
from matplotlib import pyplot as plt
import pickle
import numpy as np
from imageio import get_writer
import trimesh
import pyflex
from os import devnull
import subprocess
import os
import imageio
import OpenEXR
from Imath import PixelType
import random
import cv2
from scipy import ndimage as nd
from math import ceil
import io
from PIL import Image, ImageDraw, ImageFont
import skimage.morphology as morph
from pathlib import Path

#################################################
################# RENDER UTILS ##################
#################################################


def grid_index(x, y, dimx):
    return y*dimx + x


def get_cloth_mesh(
        dimx,
        dimy,
        base_index=0):
    if dimx == -1 or dimy == -1:
        positions = pyflex.get_positions().reshape((-1, 4))
        vertices = positions[:, :3]
        faces = pyflex.get_faces().reshape((-1, 3))
    else:
        positions = pyflex.get_positions().reshape((-1, 4))
        faces = []
        vertices = positions[:, :3]
        for y in range(dimy):
            for x in range(dimx):
                if x > 0 and y > 0:
                    faces.append([
                        base_index + grid_index(x-1, y-1, dimx),
                        base_index + grid_index(x, y-1, dimx),
                        base_index + grid_index(x, y, dimx)
                    ])
                    faces.append([
                        base_index + grid_index(x-1, y-1, dimx),
                        base_index + grid_index(x, y, dimx),
                        base_index + grid_index(x-1, y, dimx)])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def blender_render_cloth(cloth_mesh, resolution):
    output_prefix = '/tmp/' + str(os.getpid())
    obj_path = output_prefix + '.obj'
    cloth_mesh.export(obj_path)
    commands = [
        'blender',
        'cloth.blend',
        '-noaudio',
        '-E', 'BLENDER_EEVEE',
        '--background',
        '--python',
        'render_rgbd.py',
        obj_path,
        output_prefix,
        str(resolution)]
    with open(devnull, 'w') as FNULL:
        while True:
            try:
                # render images
                subprocess.check_call(
                    commands,
                    stdout=FNULL)
                break
            except Exception as e:
                print(e)
    # get images
    output_dir = Path(output_prefix)
    color = imageio.imread(str(list(output_dir.glob('*.png'))[0]))
    color = color[:, :, :3]
    depth = OpenEXR.InputFile(str(list(output_dir.glob('*.exr'))[0]))
    redstr = depth.channel('R', PixelType(PixelType.FLOAT))
    depth = np.fromstring(redstr, dtype=np.float32)
    depth = depth.reshape(resolution, resolution)
    return color, depth


def render_lift_cloth(cloth_mesh, resolution):
    output_prefix = '/tmp/' + str(os.getpid())
    obj_path = output_prefix + '.obj'
    cloth_mesh.export(obj_path)
    commands = [
        'blender',
        'lifted_cloth.blend',
        '-noaudio',
        '-E', 'BLENDER_EEVEE',
        '--background',
        '--python',
        'render_rgbd.py',
        obj_path,
        output_prefix,
        str(resolution)]
    with open(devnull, 'w') as FNULL:
        while True:
            try:
                # render images
                subprocess.check_call(
                    commands,
                    stdout=FNULL)
                break
            except Exception as e:
                print(e)
    # get images
    output_dir = Path(output_prefix)
    color = imageio.imread(str(list(output_dir.glob('*.png'))[0]))
    color = color[:, :, :3]
    depth = OpenEXR.InputFile(str(list(output_dir.glob('*.exr'))[0]))
    redstr = depth.channel('R', PixelType(PixelType.FLOAT))
    depth = np.fromstring(redstr, dtype=np.float32)
    depth = depth.reshape(resolution, resolution)
    return color, depth

#################################################
################ TRANSFORM UTILS ################
#################################################


def rot2d(angle, degrees=True):
    if degrees:
        angle = np.pi*angle/180
    return np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]).T


def translate2d(translation):
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1],
    ]).T


def scale2d(scale):
    return np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1],
    ]).T


def get_transform_matrix(original_dim, resized_dim, rotation, scale):
    # resize
    resize_mat = scale2d(original_dim/resized_dim)
    # scale
    scale_mat = np.matmul(
        np.matmul(
            translate2d(-np.ones(2)*(resized_dim//2)),
            scale2d(scale),
        ), translate2d(np.ones(2)*(resized_dim//2)))
    # rotation
    rot_mat = np.matmul(
        np.matmul(
            translate2d(-np.ones(2)*(resized_dim//2)),
            rot2d(rotation),
        ), translate2d(np.ones(2)*(resized_dim//2)))
    return np.matmul(np.matmul(scale_mat, rot_mat), resize_mat)


def compute_pose(pos, lookat, up=[0, 0, 1]):
    norm = np.linalg.norm
    if type(lookat) != np.array:
        lookat = np.array(lookat)
    if type(pos) != np.array:
        pos = np.array(pos)
    if type(up) != np.array:
        up = np.array(up)
    f = (lookat - pos)
    f = f/norm(f)
    u = up / norm(up)
    s = np.cross(f, u)
    s = s/norm(s)
    u = np.cross(s, f)
    view_matrix = [
        s[0], u[0], -f[0], 0,
        s[1], u[1], -f[1], 0,
        s[2], u[2], -f[2], 0,
        -np.dot(s, pos), -np.dot(u, pos), np.dot(f, pos), 1
    ]
    view_matrix = np.array(view_matrix).reshape(4, 4).T
    pose_matrix = np.linalg.inv(view_matrix)
    pose_matrix[:, 1:3] = -pose_matrix[:, 1:3]
    return pose_matrix


def compute_intrinsics(fov, image_size):
    image_size = float(image_size)
    focal_length = (image_size / 2)\
        / np.tan((np.pi * fov / 180) / 2)
    return np.array([[focal_length, 0, image_size / 2],
                     [0, focal_length, image_size / 2],
                     [0, 0, 1]])


def pixel_to_3d(depth_im, x, y,
                pose_matrix,
                fov=39.5978,
                depth_scale=1):
    intrinsics_matrix = compute_intrinsics(fov, depth_im.shape[0])
    click_z = depth_im[y, x]
    click_z *= depth_scale
    click_x = (x-intrinsics_matrix[0, 2]) * \
        click_z/intrinsics_matrix[0, 0]
    click_y = (y-intrinsics_matrix[1, 2]) * \
        click_z/intrinsics_matrix[1, 1]
    if click_z == 0:
        raise Exception('Invalid pick point')
    # 3d point in camera coordinates
    point_3d = np.asarray([click_x, click_y, click_z])
    point_3d = np.append(point_3d, 1.0).reshape(4, 1)
    # Convert camera coordinates to world coordinates
    target_position = np.dot(pose_matrix, point_3d)
    target_position = target_position[0:3, 0]
    target_position[0] = - target_position[0]
    return target_position


def pixels_to_3d_positions(
        pixels, scale, rotation, pretransform_depth,
        transformed_depth, pose_matrix=None,
        pretransform_pix_only=False, **kwargs):
    mat = get_transform_matrix(
        original_dim=pretransform_depth.shape[0],
        resized_dim=transformed_depth.shape[0],
        rotation=-rotation,  # TODO bug
        scale=scale)
    pixels = np.concatenate((pixels, np.array([[1], [1]])), axis=1)
    pixels = np.matmul(pixels, mat)[:, :2].astype(int)
    pix_1, pix_2 = pixels
    max_idx = pretransform_depth.shape[0]
    if (pixels < 0).any() or (pixels >= max_idx).any():
        return {
            'valid_action': False,
            'p1': None, 'p2': None,
            'pretransform_pixels': np.array([pix_1, pix_2])
        }
    if pretransform_pix_only:
        return {
            'valid_action': True,
            'pretransform_pixels': np.array([pix_1, pix_2])
        }
    # Note this order of x,y is not a bug
    x, y = pix_1
    p1 = pixel_to_3d(depth_im=pretransform_depth,
                     x=x, y=y,
                     pose_matrix=pose_matrix)
    # Same here
    x, y = pix_2
    p2 = pixel_to_3d(depth_im=pretransform_depth,
                     x=x, y=y,
                     pose_matrix=pose_matrix)
    return {
        'valid_action': p1 is not None and p2 is not None,
        'p1': p1,
        'p2': p2,
        'pretransform_pixels': np.array([pix_1, pix_2])
    }


#################################################
############ VISUALIZATION UTILS ################
#################################################

def draw_circled_lines(pixels, shape=None, img=None, thickness=1):
    if shape is not None:
        shape = list(shape) + [4]
        img = np.zeros(shape)
    else:
        assert img is not None
    left, right = pixels
    img = cv2.circle(
        img=img, center=(int(left[1]), int(left[0])),
        radius=thickness*2, color=(0, 1, 0, 1), thickness=thickness)
    img = cv2.line(
        img=img,
        pt1=(int(left[1]), int(left[0])),
        pt2=(int(right[1]), int(right[0])),
        color=(1, 1, 0, 1), thickness=thickness)
    img = cv2.circle(
        img=img, center=(int(right[1]), int(right[0])),
        radius=thickness*2, color=(1, 0, 0, 1), thickness=thickness)
    return img


def draw_circled_lines_with_arrow(
        pixels, shape=None, img=None, thickness=1):
    if shape is not None:
        shape = list(shape) + [4]
        img = np.zeros(shape)
    else:
        assert img is not None
    left, right = pixels
    img = cv2.circle(
        img=img, center=(int(left[1]), int(left[0])),
        radius=thickness*2, color=(1, 0, 1, 1), thickness=thickness)
    img = cv2.line(
        img=img,
        pt1=(int(left[1]), int(left[0])),
        pt2=(int(right[1]), int(right[0])),
        color=(1, 1, 0, 1), thickness=thickness)
    img = cv2.circle(
        img=img, center=(int(right[1]), int(right[0])),
        radius=thickness*2, color=(0, 1, 1, 1), thickness=thickness)
    direction = np.cross((left - right).tolist() +
                         [0], np.array([0, 0, 1]))[:2]
    center_start = ((left + right) / 2).astype(int)
    center_end = center_start + direction
    img = cv2.arrowedLine(
        img=img,
        pt1=(int(center_start[1]), int(center_start[0])),
        pt2=(int(center_end[1]), int(center_end[0])),
        color=(1, 0, 0, 1), thickness=thickness)
    return img


def draw_arrow(pixels, shape=None, img=None, thickness=1, color=(0, 1, 1, 1)):
    if shape is not None:
        shape = list(shape) + [4]
        img = np.zeros(shape)
    else:
        assert img is not None
    start, end = pixels
    img = cv2.arrowedLine(
        img=img,
        pt1=(int(start[1]), int(start[0])),
        pt2=(int(end[1]), int(end[0])),
        color=color, thickness=thickness)
    return img


def draw_action(action_primitive, shape, pixels, **kwargs):
    if action_primitive == 'fling':
        return draw_circled_lines(
            shape=shape, pixels=pixels, **kwargs)
    elif action_primitive == 'stretchdrag':
        return draw_circled_lines_with_arrow(
            shape=shape, pixels=pixels, **kwargs)
    elif action_primitive == 'drag':
        return draw_arrow(
            shape=shape, pixels=pixels,
            color=(1, 0, 1, 1), **kwargs)
    elif action_primitive == 'place':
        return draw_arrow(
            shape=shape, pixels=pixels,
            color=(0, 1, 1, 1), **kwargs)
    else:
        raise NotImplementedError()


def visualize_action(
    action_primitive, transformed_pixels,
    pretransform_pixels,
    rotation, scale, pretransform_depth,
    pretransform_rgb,
    transformed_rgb, value_map=None,
    all_value_maps=None,
        **kwargs):
    if value_map is None and all_value_maps is None:
        # final resized
        if pretransform_rgb.shape[0] != pretransform_rgb.shape[1]:
            pretransform_rgb = get_square_crop(pretransform_rgb.copy())
        plt.imshow(pretransform_rgb)
        action = draw_action(
            action_primitive=action_primitive,
            shape=pretransform_depth.shape[:2],
            pixels=pretransform_pixels,
            thickness=3)
        plt.imshow(action, alpha=0.9)
        plt.title(f'Final {action_primitive}')
    else:
        fig, axes = plt.subplots(1, 3)
        fig.set_figheight(3.5)
        fig.set_figwidth(9)
        for ax in axes.flatten():
            ax.axis('off')
        if value_map is not None:
            imshow = axes[0].imshow(
                value_map, cmap='jet',
                vmin=all_value_maps.min(), vmax=all_value_maps.max())
            axes[0].set_title('Value Map')
            fig.colorbar(mappable=imshow, ax=axes[0], shrink=0.8)
        else:
            axes[0].set_title('No Value Map')
        axes[1].imshow(
            np.swapaxes(np.swapaxes(transformed_rgb, 0, -1), 0, 1))
        action = draw_action(
            action_primitive=action_primitive,
            shape=transformed_rgb.shape[-2:],
            pixels=transformed_pixels)
        axes[1].imshow(action, alpha=0.9)
        axes[1].set_title(action_primitive)
        # final resized
        if pretransform_rgb.shape[0] != pretransform_rgb.shape[1]:
            pretransform_rgb = get_square_crop(pretransform_rgb.copy())
        axes[2].imshow(pretransform_rgb)
        action = draw_action(
            action_primitive=action_primitive,
            shape=pretransform_depth.shape[:2],
            pixels=pretransform_pixels,
            thickness=3)
        if action.shape[0] != action.shape[1]:
            action = get_square_crop(action.copy())
        axes[2].imshow(action, alpha=0.9)
        axes[2].set_title(f'Final {action_primitive}')
    plt.tight_layout(pad=0)
    # dump to image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    action_visualization = np.array(Image.open(buf)).astype(np.uint8)
    if value_map is not None or all_value_maps is not None:
        plt.close(fig)
    return action_visualization


def plot_before_after(group, fontsize=16, output_path=None):
    fig, (ax1, ax2) = \
        plt.subplots(1, 2, figsize=(15, 15))
    fig.set_figheight(5)
    fig.set_figwidth(9)
    ax1.axis('off')
    ax2.axis('off')

    def get_img(key):
        return np.swapaxes(np.swapaxes(np.array(group[key]), 0, -1), 0, 1)
    # Plot before
    img = get_img('pretransform_observations')
    rgb = img[:, :, :3]
    ax1.imshow(rgb)
    ax1.set_title('Before ({:.03f})'.format(
        group.attrs['preaction_coverage']
        / group.attrs['max_coverage']),
        fontsize=fontsize)

    # Plot after
    img = get_img('next_observations')
    rgb = img[:, :, :3]
    ax2.imshow(rgb)
    ax2.set_title('After ({:.03f})'.format(
        group.attrs['postaction_coverage']
        / group.attrs['max_coverage']),
        fontsize=fontsize)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)


def visualize_grasp(group, key, path_prefix, dir_path, fontsize=16, include_videos=True):
    step = int(key.split('step')[-1].split('_last')[0])
    episode_id = int(key.split('step')[0][:-1])
    output = f'<td> Episode {episode_id}, Step {step} </td><td> '

    # Plot all observations and value maps
    if 'value_maps' in group and 'all_obs' in group:
        output_path = path_prefix + '_all.png'
        output += f'<img src="{output_path}" height="256px"> </td> <td>'
        if not os.path.exists(dir_path+output_path):
            value_maps = np.array(group['value_maps'])
            fig, axes = plt.subplots(8, 12)
            axes = axes.transpose().flatten()
            fig.set_figheight(8)
            fig.set_figwidth(12)
            max_value = value_maps.max()
            min_value = value_maps.min()
            for ax, value_map in zip(axes, value_maps):
                ax.axis('off')
                ax.imshow(value_map, cmap='jet',
                          vmin=min_value, vmax=max_value)

                if (value_map == max_value).any():
                    circle = np.zeros(value_map.shape)
                    center = value_map.shape[0]//2
                    circle = cv2.circle(
                        img=circle,
                        center=(center, center),
                        radius=center,
                        color=1,
                        thickness=3)
                    ax.imshow(circle, alpha=circle, cmap='Blues')
            plt.tight_layout(pad=0)
            plt.savefig(dir_path + output_path)
            plt.close(fig)

    if 'action_visualization' in group:
        action_vis = group['action_visualization']
        output_path = path_prefix+'_action.png'
        if not os.path.exists(dir_path+output_path):
            imageio.imwrite(dir_path + output_path, action_vis)
        output += f'<img src="{output_path}" height="256px"></td>'

    if 'visualization_dir' in group.attrs and step == 0 and include_videos:
        output += '<td style="display: flex; flex-direction: row;" >'
        vis_dir_path = group.attrs['visualization_dir']
        for video_path in Path(vis_dir_path).glob('*.mp4'):
            video_path = str(video_path)
            video_path = video_path.split('/')[-2] \
                + '/' + video_path.split('/')[-1]
            output += """
            <video height=256px autoplay loop controls muted>
                <source src="{}" type="video/mp4">
            </video>
            """.format(video_path)
    else:
        output += f'<td>Step {step}'
    if 'last' in key:
        message = "No Errors"
        if ('failed_grasp' in group.attrs and
                group.attrs['failed_grasp']):
            message = "Failed Grasp"
        elif ('cloth_stuck' in group.attrs and
                group.attrs['cloth_stuck']):
            message = "Cloth Stuck"
        elif ('timed_out' in group.attrs and
                group.attrs['timed_out']):
            message = "Timed out"
        output += f':{message}'
    output += '</td><td>'

    output_path = path_prefix + '.png'
    if not os.path.exists(dir_path+output_path):
        plot_before_after(group,
                          output_path=dir_path + output_path,
                          fontsize=fontsize)
    output += f'<img src="{output_path}" height="256px"> </td>'
    if 'faces' in group and 'gripper_states' in group and 'states' in group:
        output_pkl = {
            'faces': np.array(group['faces']),
            'gripper_states': [],
            'states': [],
        }
        for k in group['gripper_states']:
            output_pkl['gripper_states'].append(
                np.array(group['gripper_states'][k]))
        for k in group['states']:
            output_pkl['states'].append(np.array(group['states'][k]))
        output_path = dir_path + path_prefix + '.pkl'
        pickle.dump(output_pkl, open(output_path, 'wb'))
        output += f'<td> {output_path} </td>'
    return output


def add_text_to_image(image, text,
                      color='rgb(255, 255, 255)', fontsize=12):
    image = Image.fromarray(image)
    ImageDraw.Draw(image).text(
        (0, 0), text,
        fill=color,
        font=ImageFont.truetype(
            "/usr/share/fonts/truetype/lato/Lato-Black.ttf", fontsize))
    return np.array(image)


def preprocess_obs(rgb, d):
    return cat((tensor(rgb).float()/255,
                tensor(d).unsqueeze(dim=2).float()),
               dim=2).permute(2, 0, 1)


def get_largest_component(arr):
    # label connected components for mask
    labeled_arr, num_components = \
        morph.label(
            arr, return_num=True,
            background=0)
    masks = [(i, (labeled_arr == i).astype(np.uint8))
             for i in range(0, num_components)]
    masks.append((
        len(masks),
        1-(np.sum(mask for i, mask in masks) != 0)))
    sorted_volumes = sorted(
        masks, key=lambda item: np.count_nonzero(item[1]),
        reverse=True)
    for i, mask in sorted_volumes:
        if arr[mask == 1].sum() == 0:
            continue
        return mask
