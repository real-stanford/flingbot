import torch
from torchvision import transforms
import h5py
from tqdm import tqdm

REWARDS_MEAN = 0.0029411377084902638
REWARDS_STD = 0.011524952525922203
REWARDS_MAX = 0.20572495126190674
REWARDS_MIN = -0.11034914070874759


class GraspDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hdf5_path: str,
                 depth_only: bool,
                 rgb_only: bool,
                 check_validity=False,
                 filter_fn=None,
                 obs_color_jitter=True,
                 is_fling_speed=False,
                 use_normalized_coverage=True,
                 **kwargs):
        assert not depth_only or not rgb_only
        self.hdf5_path = hdf5_path
        self.hdf5_path = self.hdf5_path
        self.filter_fn = filter_fn
        self.use_normalized_coverage = use_normalized_coverage
        self.rgb_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.3,
                    saturation=0.5, hue=0.5),
                transforms.ToTensor()])\
            if obs_color_jitter else lambda x: x
        self.is_fling_speed = is_fling_speed

        self.keys = self.get_keys()
        if check_validity:
            for k in tqdm(self.keys, desc='Checking validity'):
                self.check_validity(k)
            self.keys = self.get_keys()
        self.size = len(self.keys)
        self.depth_only = depth_only
        self.rgb_only = rgb_only

    def get_keys(self):
        with h5py.File(self.hdf5_path, "r") as dataset:
            keys = []
            for k in dataset:
                try:
                    group = dataset[k]
                    if self.filter_fn is None or self.filter_fn(group):
                        keys.append(k)
                except:
                    pass
            return keys

    def check_validity(self, key):
        with h5py.File(self.hdf5_path, "a") as dataset:
            group = dataset.get(key)
            if 'actions' not in group or 'observations' not in group \
                or 'postaction_coverage' not in group.attrs or \
                    'preaction_coverage' not in group.attrs:
                del dataset[key]
                return
            action = torch.tensor(group['actions']).bool()
            if len(action[action]) != 1:
                del dataset[key]
                return
            if len(torch.tensor(group['observations']).size()) == 4:
                del dataset[key]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as dataset:
            group = dataset.get(self.keys[index])
            reward = float(group.attrs['postaction_coverage']
                           - group.attrs['preaction_coverage'])
            if self.use_normalized_coverage:
                reward /= float(group.attrs['max_coverage'])
            else:
                reward = (reward - REWARDS_MIN) /\
                    (REWARDS_MAX - REWARDS_MIN)
            reward = torch.tensor(reward).float()
            if not self.is_fling_speed:
                obs = torch.tensor(group['observations'])
                action = torch.tensor(group['actions']).bool()
            else:
                obs = torch.tensor(group['fling_observations']).squeeze()
                action = torch.tensor(group.attrs['fling_speed']).bool()

            if self.rgb_only:
                obs = obs[:3, :, :]
                obs[:3, ...] = self.rgb_transform(obs[:3, ...])
            elif self.depth_only:
                obs = obs[3, :, :].unsqueeze(dim=0)

            return (obs, action, reward)
