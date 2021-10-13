from argparse import ArgumentParser
from environment import SimEnv, TaskLoader
from learning.nets import MaximumValuePolicy
from learning.utils import GraspDataset
from environment.utils import plot_before_after
from torch.utils.data import DataLoader
from filelock import FileLock
from time import time
import torch
import h5py
import os
import ray
import random
import numpy as np


def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("Dynamic Cloth Manipulation")
    parser.add_argument('--log', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--load", type=str, default=None,
                        help="path of policy to load")
    parser.add_argument('--gui', action='store_true',
                        default=False, help='Run headless or render')
    parser.add_argument('--num_processes', type=int,
                        default=16, help='How many processes to parallelize')
    parser.add_argument('--tasks', type=str,
                        default='configs_2500_train.pkl',
                        help='path to tasks pickle')
    parser.add_argument('--eval',
                        action='store_true', default=False,
                        help='Evaluation mode or training mode')
    parser.add_argument('--dump_visualizations',
                        action='store_true', default=False)

    # Optimization
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_workers', type=int, default=0)
    # Algorithm
    parser.add_argument('--batches_per_update', type=int, default=1)
    parser.add_argument('--update_frequency', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=128)
    parser.add_argument('--save_ckpt', type=int, default=512)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--action_expl_prob', type=float, default=0.0)
    parser.add_argument('--action_expl_decay', type=float, default=0.9995)
    parser.add_argument('--value_expl_prob', type=float, default=0.0)
    parser.add_argument('--value_expl_decay', type=float, default=0.995)
    parser.add_argument('--obs_color_jitter',
                        action='store_true', default=True)
    parser.add_argument('--fixed_fling_height', type=float, default=-1)
    # Network
    parser.add_argument('--depth_only', action='store_true', default=False)
    parser.add_argument('--rgb_only', action='store_true', default=True)
    parser.add_argument('--use_adaptive_scaling',
                        action='store_true', default=True,
                        help='Automatically adjust scale_factors to fit cloth')
    parser.add_argument('--use_normalized_coverage',
                        action='store_true', default=True)
    parser.add_argument('--conservative_grasp_radius', type=int,
                        default=1)
    parser.add_argument('--action_primitives',
                        choices=['fling', 'stretchdrag', 'drag', 'place'],
                        default=['fling'], nargs='+')
    parser.add_argument('--obs_dim', type=int, default=64,
                        help='H x W of observation images')
    parser.add_argument('--pix_grasp_dist', type=int, default=8,
                        help="How wide grasp is in pixel space")
    parser.add_argument('--pix_drag_dist', type=int, default=10,
                        help="How far drag is in pixel space")
    parser.add_argument('--pix_place_dist', type=int, default=10,
                        help="How far pick and place is in pixel space")
    parser.add_argument('--stretchdrag_dist', type=int, default=0.3,
                        help="How far drag is for stretchdrag primitive")
    parser.add_argument('--reach_distance_limit', type=float, default=1.2,
                        help="How far can each arm reach from its workspace")
    parser.add_argument('--num_rotations', type=int, default=12,
                        help="Number of discrete rotations between -90 and 90")
    parser.add_argument('--scale_factors', nargs='+',
                        default=[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75],
                        help="Scale factors to use")
    parser.add_argument(
        '--render_engine', choices=['blender', 'opengl'],
        help="Which backend to render cloths with.", default='blender')
    return parser


def seed_all(seed):
    print(f"SEEDING WITH {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_network(args):
    policy = MaximumValuePolicy(**vars(args))
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)
    checkpoint_path = args.load
    dataset_path = args.dataset_path

    if args.log is not None and \
            os.path.exists(args.log)\
            and checkpoint_path is None:
        if os.path.exists(f'{args.log}/latest_ckpt.pth'):
            checkpoint_path = f'{args.log}/latest_ckpt.pth'

    if checkpoint_path is not None:
        print(f'Loading checkpoint {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location=policy.device)
        policy.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print('Continuing from:')
        print(f'\tStep: {policy.steps()}')
        print(
            f'\tExploration Probability: {policy.action_expl_prob.item():.4e}')
        print(f'\tExploration Decay: {policy.action_expl_decay.item():.4e}')

    if args.eval:
        assert args.load is not None
        optimizer = None
        policy.expl_prob = torch.nn.parameter.Parameter(
            torch.tensor(0.0), requires_grad=False)
        prefix = str(args.load).split('.pth')[0]
        i = 0
        args.log = prefix + f'_eval_{i}/'
        while os.path.exists(args.log):
            i += 1
            args.log = prefix + f'_eval_{i}/'
        dataset_path = args.log + 'replay_buffer.hdf5'
        print(f"Evaluating {args.load}: saving to {dataset_path}")
    elif dataset_path is None and args.log is not None:
        dataset_path = f'{args.log}/replay_buffer.hdf5'
        print(f'Replay Buffer path: {dataset_path}')
    return policy, optimizer, dataset_path


def setup_envs(dataset, num_processes=16, **kwargs):
    task_loader = ray.remote(TaskLoader).remote(
        hdf5_path=kwargs['tasks'],
        repeat=not kwargs['eval'])

    envs = [ray.remote(SimEnv).options(
        num_gpus=torch.cuda.device_count()/num_processes,
        num_cpus=0.1).remote(
        replay_buffer_path=dataset,
        get_task_fn=lambda: ray.get(task_loader.get_next_task.remote()),
        **kwargs)
        for _ in range(num_processes)]
    ray.get([e.setup_ray.remote(e) for e in envs])
    return envs, task_loader


def get_loader(batch_size=256,
               num_workers=4,
               **kwargs):
    try:
        dataset = GraspDataset(**kwargs)
    except:
        dataset = GraspDataset(
            check_validity=True,
            **kwargs)
    if len(dataset) < batch_size:
        return None
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers)


def get_dataset_size(path, pbar=None):
    if not os.path.exists(path):
        return 0
    with FileLock(path + ".lock"):
        return len(h5py.File(path, "r"))


def collect_stats(dataset_path, num_points=128,
                  action_primitives=['fling', 'stretchdrag', 'drag', 'place'],
                  pad_episode=False, filter_keys_fn=None):
    with FileLock(dataset_path + ".lock"):
        with h5py.File(dataset_path, "r") as dataset:
            # latest keys
            keys = [k for k in dataset]
            if filter_keys_fn is not None:
                keys = [k for i, k in enumerate(keys)
                        if filter_keys_fn(i, k)]
            elif len(keys) > num_points:
                keys = keys[-num_points:]
            num_points = len(keys)
            # log statistic
            stats = {
                'delta_coverage':
                {
                    'easy': [],
                    'hard': [],
                },
                'delta_coverage_steps':
                {
                    'easy': {},
                    'hard': {},
                },
                'final_coverage':
                {
                    'easy': [],
                    'hard': [],
                },
                'init_coverage':
                {
                    'easy': [],
                    'hard': [],
                },
                'best_coverage':
                {
                    'easy': [-1],
                    'hard': [-1],
                },
                'episode_delta_coverage':
                {
                    'easy': [],
                    'hard': [],
                },
                'episode_length':
                {
                    'easy': [],
                    'hard': [],
                },
                'action_primitives_steps':
                {
                    'easy': {},
                    'hard': {},
                },
                'postaction_coverage_steps':
                {
                    'easy': {},
                    'hard': {},
                },
                'preaction_coverage_steps':
                {
                    'easy': {},
                    'hard': {},
                }
            }
            action_primitive_counts = {
                ap: 0 for ap in action_primitives}

            def find_episode_length(episode):
                for k in keys:
                    if episode in k and 'last' in k:
                        return int(k.split('step')[1].split('_')[0])
                assert False
            for k in keys:
                group = dataset.get(k)
                if ('failed_grasp' in group.attrs and
                        group.attrs['failed_grasp']) or\
                        ('cloth_stuck' in group.attrs and
                         group.attrs['cloth_stuck']) or \
                        ('timed_out' in group.attrs and
                         group.attrs['timed_out']):
                    continue

                try:
                    max_coverage = group.attrs['max_coverage']
                except:
                    continue
                if group.attrs['postaction_coverage']/max_coverage < 0.05:
                    continue
                level = str(group.attrs['task_difficulty'])
                stats['delta_coverage'][level].append(
                    (group.attrs['postaction_coverage'] -
                     group.attrs['preaction_coverage'])
                    / max_coverage)
                action_primitive = group.attrs['action_primitive']
                action_primitive_counts[action_primitive] += 1
                stats['best_coverage'][level][-1] \
                    = max(stats['best_coverage'][level][-1],
                          group.attrs['postaction_coverage'] /
                          max_coverage)
                step = k.split('step')[1].split('_')[0]
                if step not in stats['delta_coverage_steps'][level]:
                    stats['delta_coverage_steps'][level][step] = []
                stats['delta_coverage_steps'][level][step].append(
                    stats['delta_coverage'][level][-1])
                if step not in stats['postaction_coverage_steps'][level]:
                    stats['postaction_coverage_steps'][level][step] = []
                stats['postaction_coverage_steps'][level][step].append(
                    group.attrs['postaction_coverage']/max_coverage)
                if step not in stats['preaction_coverage_steps'][level]:
                    stats['preaction_coverage_steps'][level][step] = []
                stats['preaction_coverage_steps'][level][step].append(
                    group.attrs['preaction_coverage']/max_coverage)
                if step not in stats['action_primitives_steps'][level]:
                    stats['action_primitives_steps'][level][step] = {
                        ap: 0 for ap in action_primitives}
                stats['action_primitives_steps'][level][step][
                    action_primitive] += 1
                if 'last' in k:
                    stats['episode_length'][level].append(
                        int(k.split('step')[1].split('_')[0]))
                    stats['final_coverage'][level].append(
                        group.attrs['postaction_coverage']/max_coverage)
                    # print(len(stats['final_coverage'][level]), k,
                    #       f'{stats["final_coverage"][level][-1]:.04f}')
                    stats['init_coverage'][level].append(
                        group.attrs['init_coverage']/max_coverage)
                    stats['best_coverage'][level].append(-1)
                    stats['episode_delta_coverage'][level].append(
                        stats['final_coverage'][level][-1] -
                        group.attrs['init_coverage']/max_coverage)
                    if pad_episode:
                        for step_i in range(int(step), 25):
                            step_i = f'{step_i:02d}'
                            if step_i not in \
                                    stats['postaction_coverage_steps'][level]:
                                stats['postaction_coverage_steps'][level][step_i] = []
                            stats['postaction_coverage_steps'][
                                level][str(step_i)].append(
                                group.attrs['postaction_coverage']
                                    / max_coverage)
                            if step_i not in \
                                    stats['preaction_coverage_steps'][level]:
                                stats['preaction_coverage_steps'][level][step_i] = []
                            stats['preaction_coverage_steps'][
                                level][str(step_i)].append(
                                group.attrs['preaction_coverage']
                                    / max_coverage)
            del stats['best_coverage']['easy'][-1]
            del stats['best_coverage']['hard'][-1]
            # normalize proportion of action primitives
            for level_steps in stats['action_primitives_steps'].values():
                for step in level_steps:
                    total = 0
                    for ap in level_steps[step]:
                        total += level_steps[step][ap]
                    if total == 0:
                        continue
                    for ap in level_steps[step]:
                        level_steps[step][ap] /= total
            retval = {}
            for key in stats:
                if '_steps' in key:
                    retval[key] = stats[key]
                    continue
                for level in stats[key]:
                    if len(stats[key][level]) == 0:
                        continue
                    stats[key][level] = np.array(stats[key][level])
                    retval[f'{key}/{level}/distribution'] = stats[key][level]
                    retval[f'{key}/{level}/mean'] = stats[key][level].mean()
                    retval[f'{key}/{level}/max'] = stats[key][level].max()
                    retval[f'{key}/{level}/min'] = stats[key][level].min()
                    if key == 'delta_coverage':
                        retval[f'{key}/{level}/percent_positive'] = \
                            np.count_nonzero(stats[key][level] > 0.0) \
                            / len(stats[key][level])
                        retval[f'{key}/{level}/percent_negative'] = \
                            np.count_nonzero(stats[key][level] < 0.0) \
                            / len(stats[key][level])
                        retval[f'{key}/{level}/percent_zero'] = \
                            np.count_nonzero(stats[key][level] == 0.0) \
                            / len(stats[key][level])
            retval.update({
                'action_primitive/percent_fling':
                action_primitive_counts['fling']/num_points,
                'action_primitive/percent_drag':
                action_primitive_counts['drag']/num_points,
                'action_primitive/percent_place':
                action_primitive_counts['place']/num_points})
            key = random.choice(keys)
            group = dataset.get(key)
            try:
                retval.update({
                    'img_before_after':
                    np.swapaxes(np.swapaxes(
                        np.array(plot_before_after(group=group)),
                        -1, 0), 1, 2),
                    'img_action_visualization':
                    torch.tensor(
                        group['action_visualization']).permute(2, 0, 1)
                })
            except:
                pass
            return retval


def step_env(all_envs, ready_envs, ready_actions, remaining_observations):
    remaining_observations.extend([e.step.remote(a)
                                   for e, a in zip(ready_envs, ready_actions)])
    step_retval = []
    start = time()
    total_time = 0
    while True:
        ready, remaining_observations = ray.wait(
            remaining_observations, num_returns=1, timeout=0.01)
        if len(ready) == 0:
            continue
        step_retval.extend(ready)
        total_time = time() - start
        if (total_time > 0.01 and len(step_retval) > 0)\
                or len(step_retval) == len(all_envs):
            break

    observations = []
    ready_envs = []

    for obs, env_id in ray.get(step_retval):
        observations.append(obs)
        ready_envs.append(env_id['val'])

    return ready_envs, observations, remaining_observations
