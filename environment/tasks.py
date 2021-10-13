import h5py
import hashlib
try:
    from environment.flex_utils import (
        get_default_config,
        set_scene,
        center_object,
        set_to_flatten,
        wait_until_stable,
        get_current_covered_area,
        PickerPickPlace
    )
except:
    from flex_utils import (
        get_default_config,
        set_scene,
        center_object,
        set_to_flatten,
        wait_until_stable,
        get_current_covered_area,
        PickerPickPlace
    )
from copy import deepcopy
import numpy as np
import pyflex
import torch
import random
from time import sleep
from typing import List
from tqdm import tqdm
from argparse import ArgumentParser
from filelock import FileLock
from pathlib import Path
import trimesh
import ray
import os


def load_cloth(path):
    """Load .obj of cloth mesh. Only quad-mesh is acceptable!
    Return:
        - vertices: ndarray, (N, 3)
        - triangle_faces: ndarray, (S, 3)
        - stretch_edges: ndarray, (M1, 2)
        - bend_edges: ndarray, (M2, 2)
        - shear_edges: ndarray, (M3, 2)
    This function was written by Zhenjia Xu
    email: xuzhenjia [at] cs (dot) columbia (dot) edu
    website: https://www.zhenjiaxu.com/
    """
    vertices, faces = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith('v '):
            vertices.append([float(n)
                             for n in line.replace('v ', '').split(' ')])
        # Face
        elif line.startswith('f '):
            idx = [n.split('/') for n in line.replace('f ', '').split(' ')]
            face = [int(n[0]) - 1 for n in idx]
            assert(len(face) == 4)
            faces.append(face)

    triangle_faces = []
    for face in faces:
        triangle_faces.append([face[0], face[1], face[2]])
        triangle_faces.append([face[0], face[2], face[3]])

    stretch_edges, shear_edges, bend_edges = set(), set(), set()

    # Stretch & Shear
    for face in faces:
        stretch_edges.add(tuple(sorted([face[0], face[1]])))
        stretch_edges.add(tuple(sorted([face[1], face[2]])))
        stretch_edges.add(tuple(sorted([face[2], face[3]])))
        stretch_edges.add(tuple(sorted([face[3], face[0]])))

        shear_edges.add(tuple(sorted([face[0], face[2]])))
        shear_edges.add(tuple(sorted([face[1], face[3]])))

    # Bend
    neighbours = dict()
    for vid in range(len(vertices)):
        neighbours[vid] = set()
    for edge in stretch_edges:
        neighbours[edge[0]].add(edge[1])
        neighbours[edge[1]].add(edge[0])
    for vid in range(len(vertices)):
        neighbour_list = list(neighbours[vid])
        N = len(neighbour_list)
        for i in range(N - 1):
            for j in range(i+1, N):
                bend_edge = tuple(
                    sorted([neighbour_list[i], neighbour_list[j]]))
                if bend_edge not in shear_edges:
                    bend_edges.add(bend_edge)

    return np.array(vertices), np.array(triangle_faces),\
        np.array(list(stretch_edges)), np.array(
            list(bend_edges)), np.array(list(shear_edges))


def generate_randomization(
        action_tool,
        cloth_mesh_path=None,
        min_cloth_size=64,
        strict_min_edge_length=64,
        max_cloth_size=104,
        task_difficulty='hard',
        cloth_type='mesh',
        gui=False,
        **kwargs):
    # cloth size here is in number of particles
    # cloth size in meters can be computed
    # as cloth_size * particle_radius
    config = deepcopy(get_default_config())
    # sample random cloth size
    cloth_dimx = np.random.randint(min_cloth_size, max_cloth_size)
    cloth_dimy = np.random.randint(min_cloth_size, max_cloth_size)
    if cloth_dimx < strict_min_edge_length and \
            cloth_dimy < strict_min_edge_length:
        return None
    mesh_verts = np.array([])
    mesh_stretch_edges = np.array([])
    mesh_bend_edges = np.array([])
    mesh_shear_edges = np.array([])
    mesh_faces = np.array([])

    if cloth_type == 'mesh':
        cloth_dimx, cloth_dimy = -1, -1
        # sample random mesh
        assert cloth_mesh_path is not None
        path = str(random.choice(list(
            Path(cloth_mesh_path).rglob('*_processed.obj'))))
        retval = load_cloth(path)
        mesh_verts = retval[0]
        mesh_faces = retval[1]
        mesh_stretch_edges, mesh_bend_edges, mesh_shear_edges = retval[2:]
        num_particle = mesh_verts.shape[0]//3
        flattened_area = trimesh.load(path).area/2
    else:
        num_particle = cloth_dimx * cloth_dimy

    # Stretch, Bend and Shear
    stiffness = np.random.uniform(0.85, 0.95, 3)
    cloth_mass = np.random.uniform(0.2, 2.0)
    config.update({
        'cloth_pos': [0, 1, 0],
        'cloth_size': [cloth_dimx, cloth_dimy],
        'cloth_stiff': stiffness,
        'cloth_mass': cloth_mass,
        'mesh_verts': mesh_verts.reshape(-1),
        'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
        'mesh_bend_edges': mesh_bend_edges.reshape(-1),
        'mesh_shear_edges': mesh_shear_edges.reshape(-1),
        'mesh_faces': mesh_faces.reshape(-1),
    })
    config = set_scene(config)
    action_tool.reset([0., -1., 0.])

    # Start with flattened cloth
    if cloth_type == 'mesh':
        positions = pyflex.get_positions().reshape(-1, 4)
        positions[:, :3] = mesh_verts
        positions[:, 1] += 0.1
        pyflex.set_positions(positions)
        for _ in range(40):
            pyflex.step()
            if gui:
                pyflex.render()
    else:
        flattened_area = set_to_flatten(config=config)

    center_object()
    if task_difficulty == 'hard':
        # Choose random pick point on cloth
        pickpoint = random.randint(0, num_particle - 1)
        curr_pos = pyflex.get_positions()
        original_inv_mass = curr_pos[pickpoint * 4 + 3]
        # Set the mass of the pickup point to infinity so that
        # it generates enough force to the rest of the cloth
        curr_pos[pickpoint * 4 + 3] = 0
        pyflex.set_positions(curr_pos)
        # Choose random height to fix random pick point on cloth to
        pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()
        height = np.random.random(1) * 1.0 + 0.5

        # Move cloth up slowly...
        init_height = pickpoint_pos[1]
        speed = 0.005
        for j in range(int(1/speed)):
            curr_pos = pyflex.get_positions()
            curr_vel = pyflex.get_velocities()
            pickpoint_pos[1] = (height-init_height)*(j*speed) + init_height
            curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
            curr_pos[pickpoint * 4 + 3] = 0
            curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
            pyflex.set_positions(curr_pos)
            pyflex.set_velocities(curr_vel)
            pyflex.step()
            if gui:
                pyflex.render()

        # Pick up the cloth and wait to stablize
        for j in range(0, 300):
            curr_pos = pyflex.get_positions()
            curr_vel = pyflex.get_velocities()
            curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
            curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
            pyflex.set_positions(curr_pos)
            pyflex.set_velocities(curr_vel)
            pyflex.step()
            if gui:
                pyflex.render()
            if wait_until_stable(
                    max_steps=1, gui=gui, tolerance=1e-1) and j > 5:
                break

        # Reset to previous cloth parameters and drop cloth
        curr_pos = pyflex.get_positions()
        curr_pos[pickpoint * 4 + 3] = original_inv_mass
        pyflex.set_positions(curr_pos)
    elif task_difficulty == 'easy':
        # Throw a few vertices up in random directions
        for _ in range(10):
            # Choose random pick point on cloth
            pickpoint = random.randint(0, num_particle - 1)
            curr_pos = pyflex.get_positions()
            original_inv_mass = curr_pos[pickpoint * 4 + 3]
            # Set the mass of the pickup point to infinity so that
            # it generates enough force to the rest of the cloth
            curr_pos[pickpoint * 4 + 3] = 0
            start_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()
            pyflex.set_positions(curr_pos)

            # Choose large random velocity
            displacement = np.random.uniform(-0.2, 0.2, 3)
            displacement[1] = 0.2
            target_pos = start_pos + displacement
            # Apply velocity for a number of time steps
            speed = 0.01
            for j in range(int(1/speed)):
                curr_pos = pyflex.get_positions()
                curr_vel = pyflex.get_velocities()
                curr_pos[pickpoint * 4: pickpoint * 4 + 3] =\
                    (target_pos - start_pos)*(j*speed) + start_pos
                curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)
                pyflex.step()
                if gui:
                    pyflex.render()
            # Reset to previous cloth parameters
            curr_pos = pyflex.get_positions()
            curr_pos[pickpoint * 4 + 3] = original_inv_mass
            pyflex.set_positions(curr_pos)
    else:
        raise NotImplementedError()
    wait_until_stable(gui=gui)
    heights = pyflex.get_positions().reshape(-1, 4)[:, 1]
    if heights.max() > 0.4:
        # probably an error
        return None
    center_object()
    return {
        'particle_pos': pyflex.get_positions(),
        'particle_vel': pyflex.get_velocities(),
        'initial_coverage': get_current_covered_area(),
        'shape_pos': pyflex.get_shape_states(),
        'phase':  pyflex.get_phases(),
        'flatten_area': flattened_area,
        'flip_mesh': 0,
        'cloth_size': np.array([cloth_dimx, cloth_dimy]),
        'cloth_stiff': stiffness,
        'cloth_mass': cloth_mass,
        'task_difficulty': task_difficulty,
        'mesh_verts': mesh_verts.reshape(-1),
        'mesh_stretch_edges': mesh_stretch_edges.reshape(-1),
        'mesh_bend_edges': mesh_bend_edges.reshape(-1),
        'mesh_shear_edges': mesh_shear_edges.reshape(-1),
        'mesh_faces': mesh_faces.reshape(-1),
    }


def generate_tasks_helper(path: str,  gui: bool, **kwargs):
    pyflex.init(
        not gui,  # headless: bool
        gui,  # render: bool
        720, 720)  # camera dimensions: int x int
    action_tool = PickerPickPlace(
        num_picker=2,
        particle_radius=0.00625,
        picker_radius=0.05,
        picker_low=(-5, 0, -5),
        picker_high=(5, 5, 5))
    while True:
        task = generate_randomization(
            action_tool,
            gui=gui,
            **kwargs)
        if task is None:
            continue
        with FileLock(path + '.lock'):
            with h5py.File(path, 'a') as file:
                key = hashlib.sha1(f'{len(file)}'.encode()).hexdigest()
                group = file.create_group(key)
                for key, value in task.items():
                    if type(value) == float or \
                            type(value) == int or\
                            type(value) == np.float64 or\
                            type(value) == str:
                        group.attrs[key] = value
                    else:
                        group.create_dataset(
                            name=key,
                            data=value,
                            compression='gzip',
                            compression_opts=9)


class Task:
    def __init__(self,
                 name: str,
                 flatten_area: float,
                 initial_coverage: float,
                 task_difficulty: str,
                 cloth_size: List = None,
                 flip_mesh: int = 0,
                 particle_pos: np.array = [],
                 particle_vel: np.array = [],
                 shape_pos: np.array = [],
                 mesh_verts: np.array = [],
                 mesh_stretch_edges: np.array = [],
                 mesh_bend_edges: np.array = [],
                 mesh_shear_edges: np.array = [],
                 mesh_faces: np.array = [],
                 phase: np.array = [],
                 cloth_stiff: np.array = [],
                 cloth_mass: float = 0.5,
                 cloth_pos=[0, 2, 0]):
        self.name = name
        self.flatten_area = flatten_area
        self.initial_coverage = initial_coverage
        self.task_difficulty = task_difficulty
        self.cloth_mass = cloth_mass
        self.cloth_size = np.array(cloth_size)
        self.particle_pos = np.array(particle_pos)
        self.particle_vel = np.array(particle_vel)
        self.shape_pos = np.array(shape_pos)
        self.phase = np.array(phase)
        self.cloth_pos = np.array(cloth_pos)
        self.cloth_stiff = np.array(cloth_stiff)
        self.flip_mesh = flip_mesh
        self.mesh_verts = np.array(mesh_verts)
        if len(self.mesh_verts) > 0:
            self.cloth_size = np.array([-1, -1])
        self.mesh_stretch_edges = np.array(mesh_stretch_edges)
        self.mesh_bend_edges = np.array(mesh_bend_edges)
        self.mesh_shear_edges = np.array(mesh_shear_edges)
        self.mesh_faces = np.array(mesh_faces)
        cam_view = 'top_down'
        if cam_view == 'top_down':
            self.camera_pos = np.array([0, 2, 0])
            self.camera_angle = np.array([np.pi*0.5, -np.pi*0.5, 0])
        else:
            self.camera_pos = np.array([1.5, 0.5, 1.5])
            self.camera_angle = np.array([0.8, -0.1, 0])
        self.camera_width = 720
        self.camera_height = 720

    def get_config(self):
        return {
            'cloth_pos': self.cloth_pos,
            'cloth_size': self.cloth_size,
            'cloth_stiff': self.cloth_stiff,
            'cloth_mass': self.cloth_mass,
            'camera_name': 'default_camera',
            'camera_params': {
                'default_camera': {
                    'pos': self.camera_pos,
                    'angle': self.camera_angle,
                    'width': self.camera_width,
                    'height': self.camera_height,
                }
            },
            'flip_mesh': self.flip_mesh,
            'flatten_area': self.flatten_area,
            'mesh_verts': self.mesh_verts,
            'mesh_stretch_edges': self.mesh_stretch_edges,
            'mesh_bend_edges': self.mesh_bend_edges,
            'mesh_shear_edges': self.mesh_shear_edges,
            'mesh_faces': self.mesh_faces
        }

    def get_state(self):
        return {
            'particle_pos': self.particle_pos,
            'particle_vel': self.particle_vel,
            'shape_pos': self.shape_pos,
            'phase': self.phase,
            'camera_params': {
                'default_camera': {
                    'pos': self.camera_pos,
                    'angle': self.camera_angle,
                    'width': self.camera_width,
                    'height': self.camera_height,
                }
            }
        }

    def get_stats(self):
        return {
            'task_name': self.name,
            'cloth_mass': self.cloth_mass,
            'cloth_size': self.cloth_size,
            'cloth_stiff': self.cloth_stiff,
            'max_coverage': self.flatten_area,
            'task_difficulty': self.task_difficulty,
            'init_coverage': self.initial_coverage
        }

    def __str__(self):
        output = f'[Task] {self.name}\n'
        output += f'\ttask_difficulty: {self.task_difficulty}\n'
        output += '\tinitial_coverage (%): ' +\
            f'{self.initial_coverage*100/self.flatten_area:.02f}\n'
        output += f'\tcloth_mass (kg): {self.cloth_mass:.04f}\n'
        output += f'\tcloth_size: {self.cloth_size}\n'
        output += f'\tcloth_stiff: {self.cloth_stiff}\n'
        output += f'\tflatten_area (m^2): {self.flatten_area:.04f}\n'
        return output


class TaskLoader:
    def __init__(self, hdf5_path: str, repeat: bool = True):
        self.hdf5_path = hdf5_path
        self.repeat = repeat
        self.keys = None
        with h5py.File(self.hdf5_path, 'r') as tasks:
            self.keys = [key for key in tasks]
            print(f'[TaskLoader] Found {len(self.keys)} tasks from',
                  self.hdf5_path)
        self.curr_task_idx = 0

    def get_next_task(self) -> Task:
        with h5py.File(self.hdf5_path, 'r') as tasks:
            key = self.keys[self.curr_task_idx]
            group = tasks[key]
            self.curr_task_idx += 1
            if not self.repeat:
                print('[TaskLoader] {}/{}'.format(
                    self.curr_task_idx,
                    len(self.keys)))
            if self.curr_task_idx >= len(self.keys):
                if not self.repeat:
                    print('[TaskLoader] Out of tasks')
                    while True:
                        sleep(5)
                else:
                    self.curr_task_idx = 0
            return Task(name=key, **group.attrs, **group)


if __name__ == "__main__":
    parser = ArgumentParser('Task Generation')
    parser.add_argument("--path", type=str, required=True,
                        help="path to output HDF5 dataset")
    parser.add_argument("--cloth_mesh_path", type=str,
                        help="path to root dir containing cloth OBJs")
    parser.add_argument("--cloth_type", type=str, default='square',
                        choices=['square', 'mesh'],
                        help="type of cloth task to create")
    parser.add_argument("--num_tasks", type=int, default=100,
                        help="number of tasks to generate")
    parser.add_argument("--num_processes", type=int, default=8,
                        help="number of parallel environments")
    parser.add_argument("--gui", action='store_true',
                        help="Run headless or not")
    parser.add_argument("--min_cloth_size", type=int, default=64)
    parser.add_argument("--strict_min_edge_length", type=int, default=64)
    parser.add_argument("--max_cloth_size", type=int, default=104)
    args = parser.parse_args()
    ray.init()
    helper_fn = ray.remote(generate_tasks_helper).options(
        num_gpus=torch.cuda.device_count()/args.num_processes)
    handles = [helper_fn.remote(**vars(args))
               for _ in range(args.num_processes)]
    with tqdm(total=args.num_tasks,
              desc='Generating tasks',
              dynamic_ncols=True) as pbar:
        while True:
            ray.wait(handles, timeout=5)
            if not os.path.exists(args.path):
                continue
            with FileLock(args.path + '.lock'):
                with h5py.File(args.path, 'r') as file:
                    pbar.update(len(file) - pbar.n)
                    if len(file) >= args.num_tasks:
                        exit()
