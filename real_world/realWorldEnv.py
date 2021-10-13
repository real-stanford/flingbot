from real_world import (
    UR5Pair, UR5MoveTimeoutException,
    fling, stretch, pick_and_drop)
from real_world.setup import (
    get_top_cam, get_front_cam,
    CLOTHS_DATASET, CURRENT_CLOTH,
    DEFAULT_ORN, DIST_UR5, WS_PC,
    MIN_GRASP_WIDTH, MAX_GRASP_WIDTH,)
from real_world.stretch import is_cloth_grasped
from real_world.utils import (

    pix_to_3d_position, get_workspace_crop,
    get_cloth_mask, compute_coverage,
    bound_grasp_pos)
from environment.simEnv import SimEnv
from environment.Memory import Memory
from environment.tasks import Task

from learning.nets import prepare_image
from time import sleep, strftime
from real_world.realur5_utils import setup_thread
from environment.utils import (
    preprocess_obs,
    add_text_to_image)
from filelock import FileLock
from copy import deepcopy
import numpy as np
from time import time
import os
import h5py
import cv2


class GraspFailException(Exception):
    def __init__(self):
        super().__init__('Grasp failed due to real world')


class RealWorldEnv(SimEnv):
    def __init__(self, replace_background=True, **kwargs):
        self.replace_background = replace_background

        def randomize_cloth():
            pick_and_drop(
                ur5_pair=self.ur5_pair, top_camera=self.top_cam,
                top_cam_right_ur5_pose=self.top_cam_right_ur5_pose,
                top_cam_left_ur5_pose=self.top_cam_left_ur5_pose,
                cam_depth_scale=self.cam_depth_scale)
            self.ur5_pair.out_of_the_way()
            return Task(
                name=f'{CURRENT_CLOTH}' +
                strftime("%Y-%m-%d_%H-%M-%S"),
                flatten_area=CLOTHS_DATASET[CURRENT_CLOTH]['flatten_area'],
                initial_coverage=self.compute_coverage(),
                task_difficulty='hard',
                cloth_mass=CLOTHS_DATASET[CURRENT_CLOTH]['mass'],
                cloth_size=CLOTHS_DATASET[CURRENT_CLOTH]['cloth_size'],
            )
        super().__init__(
            get_task_fn=randomize_cloth,
            parallelize_prepare_image=True,
            episode_length=10,
            **kwargs)
        # state variables to handle recorder
        self.recording = False
        self.recording_daemon = None
        np.random.seed(int(time()))
        self.action_handlers = {
            'fling': self.pick_and_fling_primitive,
            'drag': self.pick_and_drag_primitive,
            'place': self.pick_and_place_primitive
        }

    def setup_env(self):
        # used for getting obs
        self.top_cam = get_top_cam()

        # used for stretching primitive
        self.front_cam = get_front_cam()

        # used for recording visualizations
        # if you have a third camera/webcam,
        # you can setup a different camera here
        # with a better view of both arms
        self.setup_cam = None

        self.ur5_pair = UR5Pair()
        self.ur5_pair.open_grippers()
        self.ur5_pair.out_of_the_way()
        self.top_cam_right_ur5_pose = np.loadtxt(
            'top_down_right_ur5_cam_pose.txt')
        self.top_cam_left_ur5_pose = np.loadtxt(
            'top_down_left_ur5_cam_pose.txt')
        self.cam_depth_scale = np.loadtxt('camera_depth_scale.txt')

    def get_cloth_mask(self, rgb=None):
        if rgb is None:
            rgb = self.top_cam.get_rgbd()[0]
        return get_cloth_mask(rgb)

    def preaction(self):
        self.preaction_mask = self.get_cloth_mask()

    def compute_iou(self):
        mask = self.get_cloth_mask()
        intersection = np.logical_and(
            mask, self.preaction_mask).sum()
        union = np.logical_or(mask, self.preaction_mask).sum()
        return intersection/union

    def postaction(self):
        iou = self.compute_iou()
        print(f'\tIoU: {iou:.04f}')
        if iou > 1 - 1e-1:
            self.terminate = True

    def step(self, value_maps):
        # NOTE: negative current_timestep's
        # act as error codes
        print(f'Step {self.current_timestep}')
        try:
            retval = super().step(value_maps)
            self.episode_memory.add_value(
                key='failed_grasp', value=0)
            self.episode_memory.add_value(
                key='timed_out', value=0)
            self.episode_memory.add_value(
                key='cloth_stuck', value=0)
            return retval
        except GraspFailException as e:
            # action failed in real world
            print('\t[ERROR]', e)
            current_timestep = self.current_timestep
            self.current_timestep = -2
            self.ur5_pair.open_grippers()
            self.ur5_pair.out_of_the_way()
            if self.dump_visualizations:
                self.stop_recording()
            self.current_timestep = current_timestep
            # remove previous observation
            del self.episode_memory.data['observations'][-1]
            self.episode_memory.data['failed_grasp'] = [
                1] * len(self.episode_memory)
            print(f'\tNow episode has {len(self.episode_memory)} steps')
            self.on_episode_end()
            return self.reset()
        except UR5MoveTimeoutException as e:
            # action failed in real world
            print('\t[ERROR]', e)
            current_timestep = self.current_timestep
            self.current_timestep = -4
            self.ur5_pair.open_grippers()
            self.ur5_pair.out_of_the_way()
            if self.dump_visualizations:
                self.stop_recording()
            self.current_timestep = current_timestep
            # remove previous observation
            del self.episode_memory.data['observations'][-1]
            self.episode_memory.data['timed_out'] = [
                1] * len(self.episode_memory)
            print(f'\tNow episode has {len(self.episode_memory)} steps')
            self.on_episode_end()
            return self.reset()

    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        self.recording_daemon = setup_thread(
            target=self.record_video_daemon_fn)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.recording_daemon.join()
        self.recording_daemon = None

    def record_video_daemon_fn(self):
        while self.recording:
            # NOTE: negative current_timestep's
            # act as error codes
            text = f'step {self.current_timestep}'
            if self.current_timestep == -1:
                text = 'randomizing cloth'
            elif self.current_timestep == -2:
                text = 'grasp failed'
            elif self.current_timestep == -3:
                text = 'cloth stuck'
            elif self.current_timestep == -4:
                text = 'ur5 timed out'
            if 'top' not in self.env_video_frames:
                self.env_video_frames['top'] = []
            top_view = cv2.resize(
                get_workspace_crop(
                    self.top_cam.get_rgbd()[0].copy()),
                (256, 256))
            self.env_video_frames['top'].append(
                add_text_to_image(
                    image=top_view,
                    text=text, fontsize=16))
            if self.setup_cam is not None:
                if 'setup' not in self.env_video_frames:
                    self.env_video_frames['setup'] = []
                self.env_video_frames['setup'].append(
                    self.setup_cam.get_rgbd(repeats=1)[0])
            if len(self.env_video_frames['setup']) > 50000:
                # episode typically ends in 4000 frames
                print('Robot probably got into error... Terminating')
                exit()

    def pick_and_fling_primitive(
            self, p1, p2,
            grasp_width,
            p1_grasp_cloth: bool,
            p2_grasp_cloth: bool,
            fling_height=0.25):
        left_point, right_point = p1, p2
        left_point = bound_grasp_pos(left_point)
        right_point = bound_grasp_pos(right_point)

        if self.dump_visualizations:
            self.start_recording()
        # grasp cloth
        self.ur5_pair.movel(
            params=[
                # left
                left_point + DEFAULT_ORN,
                # right
                right_point + DEFAULT_ORN],
            blocking=True, use_pos=True)
        self.ur5_pair.close_grippers()
        # this slow backup help with grasp
        # success in pinch grasping cloths
        left_point[-1] += 0.03
        right_point[-1] += 0.03
        self.ur5_pair.movel(
            params=[
                # left
                left_point + DEFAULT_ORN,
                # right
                right_point + DEFAULT_ORN],
            blocking=True, use_pos=True,
            j_vel=0.01, j_acc=0.01)
        self.ur5_pair.close_grippers()
        # lift cloth
        dx = (DIST_UR5 - grasp_width)/2
        self.ur5_pair.movel(
            params=[
                # left
                [dx, 0,  fling_height] + DEFAULT_ORN,
                # right
                [dx, 0,  fling_height] + DEFAULT_ORN],
            blocking=True, use_pos=True)
        left_grasping, right_grasping = is_cloth_grasped(
            depth=self.front_cam.get_rgbd()[1])
        if (p1_grasp_cloth and not right_grasping)\
                or (p2_grasp_cloth and not left_grasping):
            raise GraspFailException
        if (left_grasping or right_grasping):
            if left_grasping and right_grasping:
                grasp_width = stretch(
                    ur5_pair=self.ur5_pair,
                    front_camera=self.front_cam,
                    height=fling_height, grasp_width=grasp_width)
            left_grasping, right_grasping = is_cloth_grasped(
                depth=self.front_cam.get_rgbd()[1])
            fling(ur5_pair=self.ur5_pair,
                  height=fling_height,
                  grasp_width=grasp_width,
                  left_grasping=left_grasping,
                  right_grasping=right_grasping)
        else:
            self.terminate = True
        self.ur5_pair.open_grippers()
        self.ur5_pair.out_of_the_way()
        if self.dump_visualizations:
            self.stop_recording()

    def pick_and_drag_primitive(self, **kwargs):
        raise NotImplementedError()

    def pick_stretch_drag_primitive(self, **kwargs):
        raise NotImplementedError()

    def pick_and_place_primitive(self, p1, p2, info, height=0.2, **kwargs):
        pick_point, place_point = p1, p2
        pick_point = bound_grasp_pos(pick_point)
        place_point = bound_grasp_pos(place_point)

        arm = info['which_arm']
        should_grasp_cloth = info['should_grasp_cloth']

        if self.dump_visualizations:
            self.start_recording()
        prepick_point = deepcopy(pick_point)
        backup_point = deepcopy(pick_point)
        prepick_point[2] += 0.05
        backup_point[2] += 0.02
        preplace_point = deepcopy(place_point)
        preplace_point[2] += 0.05
        if arm == 'left':
            self.ur5_pair.left_ur5.movel(
                params=[prepick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.movel(
                params=[pick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.gripper.close(blocking=True)
            self.ur5_pair.left_ur5.movel(
                params=[backup_point + DEFAULT_ORN],
                j_vel=0.01, j_acc=0.01,
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.movel(
                params=[prepick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.movel(
                params=[preplace_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.movel(
                params=[place_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.left_ur5.gripper.open(blocking=True)
            self.ur5_pair.left_ur5.movel(
                params=[preplace_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
        elif arm == 'right':
            self.ur5_pair.right_ur5.movel(
                params=[prepick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.movel(
                params=[pick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.gripper.close(blocking=True)
            self.ur5_pair.right_ur5.movel(
                params=[backup_point + DEFAULT_ORN],
                j_vel=0.01, j_acc=0.01,
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.movel(
                params=[prepick_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.movel(
                params=[preplace_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.movel(
                params=[place_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
            self.ur5_pair.right_ur5.gripper.open(blocking=True)
            self.ur5_pair.right_ur5.movel(
                params=[preplace_point + DEFAULT_ORN],
                blocking=True, use_pos=True)
        # Lift up and see if cloth is stuck
        self.ur5_pair.move(
            move_type='l',
            params=[
                # left
                [0.5, 0.0, 0.0, *DEFAULT_ORN],
                # right
                [0.5, 0.0, 0.0, *DEFAULT_ORN]],
            blocking=True, use_pos=True)
        if should_grasp_cloth and self.compute_iou() > 0.75:
            raise GraspFailException
        self.ur5_pair.out_of_the_way()
        if self.dump_visualizations:
            self.stop_recording()

    def compute_coverage(self):
        coverage = compute_coverage(rgb=self.top_cam.get_rgbd()[0])
        print(
            f"\tCoverage: {coverage/CLOTHS_DATASET[CURRENT_CLOTH]['flatten_area']:.04f}")
        return coverage

    def get_obs(self):
        self.raw_pretransform_rgb, \
            self.raw_pretransform_depth = self.top_cam.get_rgbd()

        self.postcrop_pretransform_rgb = get_workspace_crop(
            self.raw_pretransform_rgb.copy())
        self.postcrop_pretransform_d = get_workspace_crop(
            self.raw_pretransform_depth.copy())
        w, h = self.postcrop_pretransform_d.shape
        assert w == h

        self.pretransform_rgb = cv2.resize(
            self.postcrop_pretransform_rgb,
            (256, 256))
        self.pretransform_depth = cv2.resize(
            self.postcrop_pretransform_d,
            (256, 256))
        cloth_mask = self.get_cloth_mask(self.pretransform_rgb)
        if self.replace_background:
            cloth_mask = (1-cloth_mask).astype(bool)
            self.pretransform_rgb[..., 0][cloth_mask] = 0
            self.pretransform_rgb[..., 1][cloth_mask] = 0
            self.pretransform_rgb[..., 2][cloth_mask] = 0
        x, y = np.where(cloth_mask == 1)
        dimx, dimy = self.pretransform_depth.shape
        minx = x.min()
        maxx = x.max()
        miny = y.min()
        maxy = y.max()

        self.adaptive_scale_factors = self.scale_factors.copy()
        if self.compute_coverage()/CLOTHS_DATASET[CURRENT_CLOTH]['flatten_area'] < 0.3:
            self.adaptive_scale_factors = self.adaptive_scale_factors[:4]
        if self.use_adaptive_scaling:
            try:
                # Minimum square crop
                cropx = max(dimx - 2*minx, dimx - 2*(dimx-maxx))
                cropy = max(dimy - 2*miny, dimy - 2*(dimy-maxy))
                crop = max(cropx, cropy)
                # Some breathing room
                crop = int(crop*1.5)
                if crop < dimx:
                    self.adaptive_scale_factors *= crop/dimx
                    self.episode_memory.add_value(
                        key='adaptive_scale',
                        value=float(crop/dimx))
            except Exception as e:
                print(e)
                print(self.current_task)
                exit()
        return preprocess_obs(
            self.pretransform_rgb.copy(),
            self.pretransform_depth.copy())

    def reset(self):
        self.episode_memory = Memory()
        self.episode_reward_sum = 0.
        self.current_timestep = -1
        self.terminate = False
        self.env_video_frames = {}
        if self.dump_visualizations:
            self.start_recording()
        self.current_task = self.get_task_fn()
        if self.dump_visualizations:
            self.stop_recording()
        self.current_timestep = 0
        self.init_coverage = self.compute_coverage()
        obs = self.get_obs()
        self.episode_memory.add_value(
            key='pretransform_observations', value=obs)
        self.episode_memory.add_value(
            key='failed_grasp', value=0)
        self.episode_memory.add_value(
            key='timed_out', value=0)
        self.episode_memory.add_value(
            key='cloth_stuck', value=0)
        self.transformed_obs = prepare_image(
            obs, self.get_transformations(), self.obs_dim,
            parallelize=self.parallelize_prepare_image)
        return self.transformed_obs, self.ray_handle

    def on_episode_end(self):
        self.stop_recording()
        super().on_episode_end(log=True)
        if os.path.exists(self.replay_buffer_path):
            with FileLock(self.replay_buffer_path + ".lock"):
                with h5py.File(self.replay_buffer_path, 'r') as file:
                    print('\tReplay Buffer Size:', len(file))
        print(
            '='*10 + f'EPISODE END IN {self.current_timestep} STEPS' + '='*10)

    def check_action_reachability(self, **kwargs):
        return True, None

    def get_cam_pose(self):
        return self.top_cam_right_ur5_pose

    def check_action(self, **kwargs):
        retval = super().check_action(**kwargs)
        p1, p2 = retval['pretransform_pixels'].copy()

        # need to convert from workspace pixels
        # to rectangular image pixels
        def process_pixels(pix):
            retval = pix.copy().astype(np.float32)
            ratio = self.postcrop_pretransform_d.shape[0] / \
                self.pretransform_depth.shape[0]
            retval *= ratio
            retval = retval.astype(np.uint16)
            retval[0] += WS_PC[0]
            retval[1] += WS_PC[2]
            return retval
        p1 = process_pixels(p1)
        p2 = process_pixels(p2)

        # real world safety checks
        # if a grasp fails a safety check, then set
        # that action to invalid
        cam_intr = self.top_cam.color_intr
        if kwargs['action_primitive'] == 'fling':
            try:
                # make sure they are far away from each other
                y, x = p1
                p1_grasp_cloth = self.preaction_mask[y, x]
                point_1 = list(pix_to_3d_position(
                    x=x, y=y,
                    depth_image=self.raw_pretransform_depth.copy(),
                    cam_intr=cam_intr,
                    cam_extr=self.top_cam_right_ur5_pose,
                    cam_depth_scale=self.cam_depth_scale))
                y, x = p2
                p2_grasp_cloth = self.preaction_mask[y, x]
                point_2 = list(pix_to_3d_position(
                    x=x, y=y,
                    depth_image=self.raw_pretransform_depth.copy(),
                    cam_intr=cam_intr,
                    cam_extr=self.top_cam_right_ur5_pose,
                    cam_depth_scale=self.cam_depth_scale))
                grasp_width = np.linalg.norm(
                    np.array(point_1) - np.array(point_2))
                if grasp_width < MIN_GRASP_WIDTH:
                    raise Exception(
                        f'Grasp width too small: {grasp_width:.03f}')
                if grasp_width > MAX_GRASP_WIDTH:
                    raise Exception(
                        f'Grasp width too large: {grasp_width:.03f}')
                if point_1[0] < point_2[0]:
                    # point 1 is to the right of point 2
                    left_point = list(pix_to_3d_position(
                        x=p2[1], y=p2[0],
                        depth_image=self.raw_pretransform_depth.copy(),
                        cam_intr=cam_intr,
                        cam_extr=self.top_cam_left_ur5_pose,
                        cam_depth_scale=self.cam_depth_scale))
                    right_point = point_1
                    left_grasp_cloth = p2_grasp_cloth
                    right_grasp_cloth = p1_grasp_cloth
                else:
                    # point 2 is to the right of point 1
                    left_point = list(pix_to_3d_position(
                        x=p1[1], y=p1[0],
                        depth_image=self.raw_pretransform_depth.copy(),
                        cam_intr=cam_intr,
                        cam_extr=self.top_cam_left_ur5_pose,
                        cam_depth_scale=self.cam_depth_scale))
                    right_point = point_2
                    left_grasp_cloth = p1_grasp_cloth
                    right_grasp_cloth = p2_grasp_cloth
                if not self.ur5_pair.left_ur5.check_pose_reachable(
                        pose=left_point) and \
                        self.ur5_pair.right_ur5.check_pose_reachable(
                        pose=right_point):
                    raise Exception('Point not reachable')
                if right_point[2] > 0.0 or left_point[2] > 0.0:
                    raise Exception(
                        'Grasp points too high, probably an error: ' +
                        ','.join(np.array(right_point).astype(str)) + '|' +
                        ','.join(np.array(left_point).astype(str)))
                retval.update({
                    'valid_action': True,
                    'p1': left_point,
                    'p2': right_point,
                    'grasp_width': grasp_width,
                    'p1_grasp_cloth': left_grasp_cloth,
                    'p2_grasp_cloth': right_grasp_cloth
                })
                return retval
            except Exception as e:
                print('\tBad Grasp Candidate:', e)
                return {'valid_action': False}
        # if not a fling then can just return
        return retval
