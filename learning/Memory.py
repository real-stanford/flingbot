import h5py
from filelock import FileLock
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random


class Memory:
    log = False
    base_keys = [
        'observations',
        'actions',
        'rewards',
        'is_terminal',
    ]

    def __init__(self, memory_fields=[]):
        self.data = {}
        for key in Memory.base_keys:
            self.data[key] = []
        for memory_field in memory_fields:
            self.data[memory_field] = []

    @staticmethod
    def concat(memories):
        output = Memory()
        for memory in memories:
            for key in memory.data:
                if key not in output.data:
                    output.data[key] = []
                output.data[key].extend(memory.data[key])
        return output

    def clear(self):
        for key in self.data:
            del self.data[key][:]

    def print_length(self):
        output = "[Memory] "
        for key in self.data:
            output += f" {key}: {len(self.data[key])} |"
        print(output)

    def assert_length(self):
        key_lens = [len(self.data[key]) for key in self.data]

        same_length = key_lens.count(key_lens[0]) == len(key_lens)
        if not same_length:
            self.print_length()

    def __len__(self):
        return len(self.data['observations'])

    def add_rewards_and_termination(self, reward, termination):
        assert len(self.data['rewards']) \
            == len(self.data['is_terminal'])\
            == len(self.data['actions']) - 1\
            == len(self.data['observations']) - 1
        self.data['rewards'].append(float(reward))
        self.data['is_terminal'].append(float(termination))

    def add_observation(self, observation):
        assert len(self.data['rewards']) \
            == len(self.data['is_terminal'])\
            == len(self.data['actions'])\
            == len(self.data['observations'])
        self.data['observations'].append(deepcopy(observation))

    def add_action(self, action):
        assert len(self.data['rewards']) \
            == len(self.data['is_terminal'])\
            == len(self.data['actions'])\
            == len(self.data['observations']) - 1
        self.data['actions'].append(deepcopy(action))

    def add_value(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(deepcopy(value))

    def keys(self):
        return [key for key in self.data]

    def count(self):
        return len(self.data['observations'])

    def done(self):
        if len(self.data['is_terminal']) == 0:
            return False
        return self.data['is_terminal'][-1]

    def get_data(self):
        return self.data

    def check_error(self):
        try:
            count = len(self)
            assert len(self.data['max_coverage']) == count
            assert len(self.data['preaction_coverage']) == count
            assert len(self.data['postaction_coverage']) == count
            return True
        except:
            return False

    def dump(self, hdf5_path, log=False):
        if len(self) < 1:
            return
        with FileLock(hdf5_path + ".lock"):
            with h5py.File(hdf5_path, 'a') as file:
                last_key = None
                for last_key in file:
                    pass
                key_idx = int(last_key.split('_')[0])\
                    if last_key is not None else 0
                while True:
                    group_key = f'{key_idx:09d}'
                    if (group_key + '_step00') not in file\
                            and (group_key + '_step00_last') not in file:
                        break
                    key_idx += 1
                for step in range(len(self)):
                    step_key = group_key + f'_step{step:02d}'
                    if step == len(self) - 1:
                        step_key += '_last'
                    try:
                        group = file.create_group(step_key)
                    except Exception as e:
                        print(e, step_key)
                        group = file.create_group(
                            step_key + '_' +
                            str(random.randint(0, int(1e5))))
                    for key, value in self.data.items():
                        try:
                            if any(key == skip_key
                                   for skip_key in
                                   ['visualization_dir', 'faces',
                                    'gripper_states', 'states']) \
                                    and step != 0:
                                continue
                            step_value = value[step]
                            if type(step_value) == float\
                                    or type(step_value) == np.float64\
                                    or type(step_value) == str\
                                    or type(step_value) == int:
                                group.attrs[key] = step_value
                            elif type(step_value) == list:
                                subgroup = group.create_group(key)
                                for i, item in enumerate(step_value):
                                    subgroup.create_dataset(
                                        name=f'{i:09d}',
                                        data=item,
                                        compression='gzip',
                                        compression_opts=9)
                            else:
                                group.create_dataset(
                                    name=key,
                                    data=step_value,
                                    compression='gzip',
                                    compression_opts=9)
                        except Exception as e:
                            if log:
                                print(f'[Memory] Dump key {key} error:', e)
                                print(value)
                return group_key
