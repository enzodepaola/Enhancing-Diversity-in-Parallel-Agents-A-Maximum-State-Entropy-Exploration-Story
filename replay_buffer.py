import datetime
import io
import random
import traceback
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1

def episode_counter(episode):
    return episode.shape[0]

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def relable_episode(env, episode):
    rewards = []
    reward_spec = env.reward_spec()
    states = episode['physics']
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode['reward'] = np.array(rewards, dtype=reward_spec.dtype)
    return episode


class OfflineReplayBuffer(IterableDataset):
    def __init__(self, file_path, max_size, num_workers, discount):
        self._file_path = file_path
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._loaded = False
        self.OBS = 0
        self.ACT = 1
        self.REW = 2
        self.TRUNC = 3
        self.DONE = 4
        self.NEXT_OBS = 5


    def _load(self, relable=False):
        """
        Load a specific file instead of iterating through a folder.
        
        Args:
            file_path (str or Path): The path to the specific file to load.
            relable (bool): Whether to relabel the data (default: False).
        """
        print('Labeling data...')

        # Ensure file_path is a Path object
        
        # Check if the file exists
        if not self._file_path.exists():
            raise FileNotFoundError(f"The specified file '{self._file_path}' does not exist.")
        
        # Process the specific file
        eps_fn = self._file_path
        eps_seed,eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]

        # Load the episode data
        episode = load_episode(eps_fn)
        self._episode_fns.append(eps_fn)
        self._episodes[eps_fn] = episode
        self._size += episode_counter(episode["sars"])

    def _sample_episode(self):
        if not self._loaded:
            self._load()
            self._loaded = True
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]


    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        episode_idx = np.random.randint(0, self._size)
       
        obs = episode['sars'][episode_idx,self.OBS].astype(np.int32).reshape(1,)
        action = episode['sars'][episode_idx,self.ACT].astype(np.int32).reshape(1,)
        next_obs = episode['sars'][episode_idx,self.NEXT_OBS].astype(np.int32).reshape(1,)
        reward = episode['sars'][episode_idx,self.REW].astype(np.float32).reshape(1,)
        truncated = episode['sars'][episode_idx,self.TRUNC].astype(bool).reshape(1,)
        done = episode['sars'][episode_idx,self.DONE].astype(bool).reshape(1,)
        discount = np.float32(self._discount)
        return (obs, action, reward, next_obs,truncated,done)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id


def make_replay_loader(file_path, max_size, batch_size, num_workers,
                       discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(file_path, max_size_per_worker,
                                   num_workers, discount)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader