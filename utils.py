import numpy as np
import torch

import os
import random
import imageio
import gym
from tqdm import trange
import pickle


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.max_size = max_size

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def normalize_states(self, eps=1e-3):
        min_state = np.min(self.state, 0)
        max_state = np.max(self.state, 0)
        self.state = 2 * (self.state - min_state) / (max_state - min_state + eps) - 1
        self.next_state = 2 * (self.next_state - min_state) / (max_state - min_state + eps) - 1
        return min_state, max_state

    def initialize(self, dataset):
        if dataset['observations'].shape[0] >= self.max_size:
            self.state = dataset['observations'][:self.max_size,:]
            self.action = dataset['actions'][:self.max_size,:]
            self.next_state = dataset['next_observations'][:self.max_size,:]
            self.reward = dataset['rewards'].reshape(-1,1)[:self.max_size,:]
            self.not_done = 1. - dataset['terminals'].reshape(-1,1)[:self.max_size,:]
            self.size = self.state.shape[0]
        else:
            size = dataset['observations'].shape[0]
            self.state[:size,:] = dataset['observations']
            self.action[:size,:] = dataset['actions']
            self.next_state[:size,:] = dataset['next_observations']
            self.reward[:size,:] = dataset['rewards'].reshape(-1,1)
            self.not_done[:size,:] = 1. - dataset['terminals'].reshape(-1,1)
            self.size = self.state.shape[0]
            self.ptr = self.size


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def snapshot_src(src, target, exclude_from):
    make_dir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
