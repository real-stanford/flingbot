"""
All the functions in this file has been taken from Softgym
with minimal to no modification. Please consider checking out
their work as well!

Github: https://github.com/Xingyu-Lin/softgym
Website: https://sites.google.com/view/softgym/home
Paper: https://arxiv.org/abs/2011.07215
"""

import pyflex
from copy import deepcopy
import numpy as np
import abc
import scipy.spatial
import cv2


class ActionToolBase(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self, state):
        """ Reset """

    @abc.abstractmethod
    def step(self, action):
        """ 
        Step funciton to change the action space states.
        Does not call pyflex.step()
        """


class Picker(ActionToolBase):
    def __init__(self, num_picker=1,
                 picker_radius=0.02,
                 init_pos=(0., -0.1, 0.),
                 picker_threshold=0.005,
                 particle_radius=0.05,
                 picker_low=(-0.4, 0., -0.4),
                 picker_high=(0.4, 0.5, 0.4),
                 init_particle_pos=None,
                 spring_coef=1.2, **kwargs):
        """
        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        super(Picker).__init__()
        self.picker_radius = picker_radius
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picked_particles = [None] * self.num_picker
        self.picker_low, self.picker_high = np.array(
            list(picker_low)), np.array(list(picker_high))
        self.init_pos = init_pos
        self.particle_radius = particle_radius
        self.init_particle_pos = init_particle_pos
        # Prevent picker to drag two particles too far away
        self.spring_coef = spring_coef

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.
        pos = []
        for i in range(self.num_picker):
            x = center[0] + np.cos(2 * np.pi * i / self.num_picker) * r
            y = center[1]
            z = center[2] + np.sin(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, center):
        for i in (0, 2):
            offset = center[i] - (self.picker_high[i] +
                                  self.picker_low[i]) / 2.
            self.picker_low[i] += offset
            self.picker_high[i] += offset
        init_picker_poses = self._get_centered_picker_pos(center)

        for picker_pos in init_picker_poses:
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])
        # Need to call this to update the shape collision
        pos = pyflex.get_shape_states()
        pyflex.set_shape_states(pos)

        self.picked_particles = [None] * self.num_picker
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        centered_picker_pos = self._get_centered_picker_pos(center)
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack(
                [centered_picker_pos,
                 centered_picker_pos,
                 [1, 0, 0, 0],
                 [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        # Remove this as having an additional step here
        # may affect the cloth drop env
        self.particle_inv_mass = \
            pyflex.get_positions().reshape(-1, 4)[:, 3]

    @staticmethod
    def _get_pos():
        """ 
        Get the current pos of the pickers and the particles,
         along with the inverse mass of each particle
        """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        return picker_pos[:, :3], particle_pos

    @staticmethod
    def _set_pos(picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    def step(self, action):
        """ action = [translation, pick/unpick] * num_pickers.
        1. Determine whether to pick/unpick the particle and which one,
           for each picker
        2. Update picker pos
        3. Update picked particle pos
        """
        action = np.reshape(action, [-1, 4])
        pick_flag = action[:, 3] > 0.5
        picker_pos, particle_pos = self._get_pos()
        new_particle_pos = particle_pos.copy()
        new_picker_pos = picker_pos.copy()

        # Un-pick the particles
        for i in range(self.num_picker):
            if not pick_flag[i] and self.picked_particles[i] is not None:
                # Revert the mass
                new_particle_pos[self.picked_particles[i], 3] = \
                    self.particle_inv_mass[self.picked_particles[i]]
                self.picked_particles[i] = None

        # Pick new particles and update the mass and the positions
        for i in range(self.num_picker):
            new_picker_pos[i, :] = picker_pos[i, :] + action[i, :3]
            if pick_flag[i]:
                # No particle is currently picked and
                # thus need to select a particle to pick
                if self.picked_particles[i] is None:
                    dists = scipy.spatial.distance.cdist(picker_pos[i].reshape(
                        (-1, 3)), particle_pos[:, :3].reshape((-1, 3)))
                    idx_dists = np.hstack(
                        [np.arange(particle_pos.shape[0]).reshape((-1, 1)),
                         dists.reshape((-1, 1))])
                    mask = dists.flatten() <= self.picker_threshold + \
                        self.picker_radius + self.particle_radius
                    idx_dists = idx_dists[mask, :].reshape((-1, 2))
                    if idx_dists.shape[0] > 0:
                        pick_id, pick_dist = None, None
                        for j in range(idx_dists.shape[0]):
                            if idx_dists[j, 0] not in self.picked_particles\
                                    and (pick_id is None or
                                         idx_dists[j, 1] < pick_dist):
                                pick_id = idx_dists[j, 0]
                                pick_dist = idx_dists[j, 1]
                        if pick_id is not None:
                            self.picked_particles[i] = int(pick_id)

                if self.picked_particles[i] is not None:
                    new_particle_pos[self.picked_particles[i], :3] =\
                        particle_pos[self.picked_particles[i], :3]\
                        + new_picker_pos[i, :] - picker_pos[i, :]
                    # Set the mass to infinity
                    new_particle_pos[self.picked_particles[i], 3] = 0

        # check for e.g., rope, the picker is not dragging the particles
        # too far away that violates the actual physicals constraints.
        if self.init_particle_pos is not None:
            picked_particle_idices = []
            active_picker_indices = []
            for i in range(self.num_picker):
                if self.picked_particles[i] is not None:
                    picked_particle_idices.append(self.picked_particles[i])
                    active_picker_indices.append(i)

            for i in range(len(picked_particle_idices)):
                for j in range(i + 1, len(picked_particle_idices)):
                    init_distance = np.linalg.norm(
                        self.init_particle_pos[picked_particle_idices[i], :3] -
                        self.init_particle_pos[picked_particle_idices[j], :3])
                    now_distance = np.linalg.norm(
                        new_particle_pos[picked_particle_idices[i], :3] -
                        new_particle_pos[picked_particle_idices[j], :3])
                    # if dragged too long, make the action has no effect;
                    # revert it
                    if now_distance >= init_distance * self.spring_coef:
                        new_picker_pos[active_picker_indices[i], :] = \
                            picker_pos[active_picker_indices[i], :].copy()
                        new_picker_pos[active_picker_indices[j], :] = \
                            picker_pos[active_picker_indices[j], :].copy()
                        new_particle_pos[picked_particle_idices[i], :3] =\
                            particle_pos[picked_particle_idices[i], :3].copy()
                        new_particle_pos[picked_particle_idices[j], :3] =\
                            particle_pos[picked_particle_idices[j], :3].copy()

        self._set_pos(new_picker_pos, new_particle_pos)


class PickerPickPlace(Picker):
    def __init__(self,
                 num_picker,
                 picker_low=None,
                 picker_high=None,
                 steps_limit=1,
                 **kwargs):
        super().__init__(num_picker=num_picker,
                         picker_low=picker_low,
                         picker_high=picker_high,
                         **kwargs)
        picker_low, picker_high = list(picker_low), list(picker_high)
        self.delta_move = 1.0
        self.steps_limit = steps_limit

    def step(self, action, step_sim_fn=lambda: pyflex.step()):
        """
        action: Array of pick_num x 4. For each picker,
         the action should be [x, y, z, pick/drop]. 
        The picker will then first pick/drop, and keep
         the pick/drop state while moving towards x, y, x.
        """
        total_steps = 0
        action = action.reshape(-1, 4)
        curr_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)[:, :3]
        end_pos = np.vstack([picker_pos
                             for picker_pos in action[:, :3]])
        dist = np.linalg.norm(curr_pos - end_pos, axis=1)
        num_step = np.max(np.ceil(dist / self.delta_move))
        if num_step < 0.1:
            return
        delta = (end_pos - curr_pos) / num_step
        norm_delta = np.linalg.norm(delta)
        for i in range(int(min(num_step, self.steps_limit))):
            curr_pos = np.array(pyflex.get_shape_states()
                                ).reshape(-1, 14)[:, :3]
            dist = np.linalg.norm(end_pos - curr_pos, axis=1)
            if np.alltrue(dist < norm_delta):
                delta = end_pos - curr_pos
            super().step(np.hstack([delta, action[:, 3].reshape(-1, 1)]))
            step_sim_fn()
            total_steps += 1
            if np.alltrue(dist < self.delta_move):
                break
        return total_steps


def vectorized_meshgrid(vec_x, vec_y):
    """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
    N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
    vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
    vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
    return vec_x, vec_y


def vectorized_range(start, end):
    """  Return an array of NxD, iterating from the start to the end"""
    N = int(np.max(end - start)) + 1
    idxes = np.floor(np.arange(N) * (end - start)
                     [:, None] / N + start[:, None]).astype('int')
    return idxes


def get_default_config(
        particle_radius=0.00625,
        camera_width=720,
        camera_height=720):
    cam_pos = np.array([0, 2, 0])
    cam_angle = np.array([np.pi*0.5, -np.pi*0.5, 0])
    config = {
        'cloth_pos': [-1.6, 2.0, -0.8],
        'cloth_size': [int(0.6 / particle_radius),
                       int(0.368 / particle_radius)],
        'cloth_stiff': [0.8, 1, 0.9],  # Stretch, Bend and Shear
        'camera_name': 'default_camera',
        'camera_params': {'default_camera':
                          {'pos': cam_pos,
                           'angle': cam_angle,
                           'width': camera_width,
                           'height': camera_height}},
        'flip_mesh': 0
    }

    return config


def update_camera(camera_params, camera_name='default_camera'):
    camera_param = camera_params[camera_name]
    pyflex.set_camera_params(
        np.array([
            *camera_param['pos'],
            *camera_param['angle'],
            camera_param['width'],
            camera_param['height']]))


def set_state(state_dict):
    pyflex.set_positions(state_dict['particle_pos'])
    pyflex.set_velocities(state_dict['particle_vel'])
    pyflex.set_shape_states(state_dict['shape_pos'])
    pyflex.set_phases(state_dict['phase'])
    camera_params = deepcopy(state_dict['camera_params'])
    update_camera(camera_params=camera_params)


def center_object(step_sim_fn=lambda: pyflex.step()):
    pos = pyflex.get_positions().reshape(-1, 4)
    pos[:, [0, 2]] -= np.mean(pos[:, [0, 2]], axis=0, keepdims=True)
    pyflex.set_positions(pos.flatten())
    step_sim_fn()


def set_scene(config,
              state=None,
              render_mode='cloth',
              step_sim_fn=lambda: pyflex.step()):
    if render_mode == 'particle':
        render_mode = 1
    elif render_mode == 'cloth':
        render_mode = 2
    elif render_mode == 'both':
        render_mode = 3
    camera_params = config['camera_params'][config['camera_name']]
    env_idx = 0 if 'env_idx' not in config else config['env_idx']
    scene_params = np.array([
        *config['cloth_pos'],
        *config['cloth_size'],
        *config['cloth_stiff'],
        render_mode,
        *camera_params['pos'][:],
        *camera_params['angle'][:],
        camera_params['width'],
        camera_params['height'],
        config['cloth_mass'],
        config['flip_mesh']])
    pyflex.set_scene(
        scene_idx=env_idx,
        scene_params=scene_params,
        vertices=config['mesh_verts'],
        stretch_edges=config['mesh_stretch_edges'],
        bend_edges=config['mesh_bend_edges'],
        shear_edges=config['mesh_shear_edges'],
        faces=config['mesh_faces'],
        thread_idx=0)
    step_sim_fn()
    if state is not None:
        set_state(state)
    return deepcopy(config)


def get_current_covered_area(cloth_particle_radius: float = 0.00625, pos=None):
    """
    Calculate the covered area by taking max x,y cood and min x,y 
    coord, create a discritized grid between the points
    :param pos: Current positions of the particle states
    """
    if pos is None:
        pos = pyflex.get_positions()
    pos = np.reshape(pos, [-1, 4])
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 2])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 2])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.
    pos2d = pos[:, [0, 2]]

    offset = pos2d - init
    slotted_x_low = np.maximum(
        np.round((offset[:, 0] - cloth_particle_radius) /
                 span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round(
        (offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
    slotted_y_low = np.maximum(
        np.round((offset[:, 1] - cloth_particle_radius) /
                 span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round(
        (offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
    # Method 1
    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    return np.sum(grid) * span[0] * span[1]


def set_to_flatten(config, cloth_particle_radius=0.00625):
    cloth_dimx, cloth_dimz = config['cloth_size']
    N = cloth_dimx * cloth_dimz
    px = np.linspace(
        0, cloth_dimx * cloth_particle_radius, cloth_dimx)
    py = np.linspace(
        0, cloth_dimz * cloth_particle_radius, cloth_dimz)
    xx, yy = np.meshgrid(px, py)
    new_pos = np.empty(shape=(N, 4), dtype=np.float)
    new_pos[:, 0] = xx.flatten()
    new_pos[:, 1] = cloth_particle_radius
    new_pos[:, 2] = yy.flatten()
    new_pos[:, 3] = 1.
    new_pos[:, :3] -= np.mean(new_pos[:, :3], axis=0)
    pyflex.set_positions(new_pos.flatten())
    return get_current_covered_area(
        cloth_particle_radius=cloth_particle_radius,
        pos=new_pos)


def get_image(width=720, height=720):
    rgb, depth = pyflex.render()
    # Need to reverse the height dimension
    rgb = np.flip(rgb.reshape([720, 720, 4]), 0)[:, :, :3].astype(np.uint8)
    depth = np.flip(depth.reshape([720, 720]), 0)
    if (width != rgb.shape[0] or height != rgb.shape[1]) and \
            (width is not None and height is not None):
        rgb = cv2.resize(rgb, (width, height))
        depth = cv2.resize(depth, (width, height))
    return rgb, depth


def wait_until_stable(max_steps=300,
                      tolerance=1e-2,
                      gui=False,
                      step_sim_fn=lambda: pyflex.step()):
    for _ in range(max_steps):
        particle_velocity = pyflex.get_velocities()
        if np.abs(particle_velocity).max() < tolerance:
            return True
        step_sim_fn()
        if gui:
            pyflex.render()
    return False
