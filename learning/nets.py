import torch.nn as nn
import torch
from scipy import ndimage as nd
import cv2
from typing import List
import random
from time import time
import ray
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes,
                 kernel_size, stride,
                 padding=1,
                 norm_layer=None,
                 non_linearity=nn.LeakyReLU):
        super(BasicBlock, self).__init__()
        if non_linearity is not None:
            self.net = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=False),
                nn.BatchNorm2d(planes),
                non_linearity()
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(inplanes,
                          planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=False)
            )

    def forward(self, input):
        return self.net(input)


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes,
                 kernel_size, stride,
                 norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.planes = planes
        self.stride = stride
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class SpatialValueNet(nn.Module):
    def __init__(self, rgb_only=False, depth_only=False,
                 steps=0, device='cuda', **kwargs):
        super().__init__()
        self.input_channels = 4
        self.device = device
        if rgb_only:
            self.input_channels = 3
        elif depth_only:
            self.input_channels = 1
        self.net = self.setup_net()
        self.rgb_only = rgb_only
        self.depth_only = depth_only
        self.mean = torch.tensor([0.18, 0.18, 0.18, 1.99])
        self.std = torch.tensor([0.1, 0.1, 0.1, 0.006])
        if self.rgb_only:
            self.mean = self.mean[:3]
            self.std = self.std[:3]
        elif self.depth_only:
            self.mean = self.mean[3]
            self.std = self.std[3]
        self.steps = nn.parameter.Parameter(
            torch.tensor(steps), requires_grad=False)

    def setup_net(self):
        return nn.Sequential(
            BasicBlock(self.input_channels, 16, 3, 1),
            #
            ResidualBlock(16, 16, 3, 1),
            ResidualBlock(16, 16, 3, 1),
            ResidualBlock(16, 16, 3, 1),
            ResidualBlock(16, 16, 3, 1),
            #
            ResidualBlock(16, 16, 3, 1),
            ResidualBlock(16, 16, 3, 1),
            ResidualBlock(16, 16, 3, 1),
            ResidualBlock(16, 16, 3, 1),
            #
            BasicBlock(16, 1, 3, 1, non_linearity=None)
        )

    def preprocess_obs(self, obs):
        assert len(obs.size()) == 4
        b, c, h, w = obs.shape
        mean = self.mean.to(obs.device)
        std = self.std.to(obs.device)
        if self.rgb_only:
            if c == 4:
                obs = obs[:, :3, :, :]
            elif c != 3:
                raise Exception
        elif self.depth_only:
            if c == 4:
                obs = obs[:, 3, :, :].unsqueeze(dim=1)
            else:
                obs = obs.squeeze().unsqueeze(dim=-3)
        obs = (obs.permute(0, 2, 3, 1) - mean) / std
        return obs.permute(0, 3, 1, 2)

    def forward(self, obs):
        return self.net(self.preprocess_obs(obs))


def crop_center(img, crop):
    startx = img.shape[1]//2-(crop//2)
    starty = img.shape[0]//2-(crop//2)
    return img[starty:starty+crop, startx:startx+crop, ...]


def pad(img, size):
    n = (size-img.shape[0])//2
    return cv2.copyMakeBorder(img, n, n, n, n, cv2.BORDER_REPLICATE)


def transform(img, rotation: float, scale: float, dim: int):
    if len(img.shape) == 3 and (img.shape[-1] == img.shape[-2]):
        img = img.permute(2, 1, 0)
    # rotate
    img = nd.rotate(input=img,
                    angle=rotation,
                    reshape=False,
                    mode='nearest')
    # scale
    new_dim = int(scale*img.shape[0])
    if scale < 1:
        img = crop_center(img, new_dim)
    elif scale > 1:
        img = pad(img, new_dim)
    # resize
    img = cv2.resize(img, dsize=(dim, dim),
                     interpolation=cv2.INTER_NEAREST)
    if len(img.shape) == 3:
        img = img.swapaxes(-1, 0)
    return torch.tensor(img)


transform_async = ray.remote(transform)


def prepare_image(img, transformations, dim: int,
                  parallelize=False, log=False):
    if log:
        start = time()
        print('preparing images')
    if parallelize:
        imgs = ray.get([transform_async.remote(img, *t, dim=dim)
                        for t in transformations])
    else:
        imgs = [transform(img, *t, dim=dim) for t in transformations]
    retval = torch.stack(imgs).float()
    if log:
        print(f'prepare_image took {float(time()-start):.02f}s')
    return retval


class Policy:
    def __init__(self,
                 action_primitives: List[str],
                 num_rotations: int,
                 scale_factors: List[float],
                 obs_dim: int,
                 pix_grasp_dist: int,
                 pix_drag_dist: int,
                 pix_place_dist: int,
                 **kwargs):
        assert len(action_primitives) > 0
        self.action_primitives = action_primitives
        print('[Policy] Action primitives:')
        for ap in self.action_primitives:
            print(f'\t{ap}')

        # rotation angle in degrees, counter-clockwise
        self.rotations = [(2*i/(num_rotations-1) - 1) * 90
                          for i in range(num_rotations)]
        if 'fling' not in action_primitives:
            self.rotations = [(2*i/num_rotations - 1) *
                              180 for i in range(num_rotations)]
        self.scale_factors = scale_factors
        self.num_transforms = len(self.rotations) * len(self.scale_factors)
        self.obs_dim = obs_dim
        self.pix_grasp_dist = pix_grasp_dist
        self.pix_drag_dist = pix_drag_dist
        self.pix_place_dist = pix_place_dist

    def get_action_single(self, obs):
        raise NotImplementedError()

    def act(self, obs):
        return [self.get_action_single(o) for o in obs]


class MaximumValuePolicy(nn.Module, Policy):
    def __init__(self,
                 action_expl_prob: float,
                 action_expl_decay: float,
                 value_expl_prob: float,
                 value_expl_decay: float,
                 device=None,
                 **kwargs):
        super().__init__()
        Policy.__init__(self, **kwargs)
        if device is None:
            self.device = torch.device('cuda') \
                if torch.cuda.is_available()\
                else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.action_expl_prob = nn.parameter.Parameter(
            torch.tensor(action_expl_prob), requires_grad=False)
        self.action_expl_decay = nn.parameter.Parameter(
            torch.tensor(action_expl_decay), requires_grad=False)
        self.value_expl_prob = nn.parameter.Parameter(
            torch.tensor(value_expl_prob), requires_grad=False)
        self.value_expl_decay = nn.parameter.Parameter(
            torch.tensor(value_expl_decay), requires_grad=False)

        # one value net per action primitive
        self.value_nets = nn.ModuleDict({key: SpatialValueNet(
            device=self.device, **kwargs).to(self.device)
            for key in self.action_primitives})
        self.should_explore_action = lambda: \
            self.action_expl_prob > random.random()
        self.should_explore_value = lambda: \
            self.value_expl_prob > random.random()

        self.eval()

    def decay_exploration(self):
        self.action_expl_prob *= self.action_expl_decay
        self.value_expl_prob *= self.value_expl_decay

    def random_value_map(self):
        return torch.rand(len(self.rotations) * len(self.scale_factors),
                          self.obs_dim, self.obs_dim)

    def get_action_single(self, obs):
        with torch.no_grad():
            obs = obs.to(self.device, non_blocking=True)
            value_maps = {key: val_net(obs).cpu().squeeze()
                          if not self.should_explore_value()
                          else self.random_value_map()

                          for key, val_net in self.value_nets.items()}
            if self.should_explore_action():
                random_action, action_val_map = random.choice(
                    list(value_maps.items()))
                min_val = action_val_map.min()
                value_maps = {
                    key: (val_map
                          if key == random_action
                          else torch.ones(val_map.size()) * min_val)
                    for key, val_map in value_maps.items()}
            return value_maps

    def steps(self):
        return sum([net.steps for net in self.value_nets.values()])

    def forward(self, obs):
        return self.act(obs)
