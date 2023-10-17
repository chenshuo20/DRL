import os
import glob
import torch
import shutil
import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from moviepy.editor import VideoFileClip, concatenate_videoclips


def moving_average(a, n):
    """
    Return an array of the moving average of a with window size n
    """
    if len(a) <= n:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_epsilon(step, eps_min, eps_max, eps_steps):
    """
    Return the linearly descending epsilon of the current step for the epsilon-greedy policy. After eps_steps, epsilon will keep at eps_min
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    if step < eps_steps:
        eps = (eps_min - eps_max) / eps_steps * step + eps_max
    else: 
        eps = eps_min
    return eps
    ############################

def merge_videos(video_dir):
    """
    Merge videos in the video_dir into a single video    
    """
    videos = glob.glob(os.path.join(video_dir, "*.mp4"))
    videos = sorted(videos, key=lambda x: int(x.split("-")[-1].split(".")[0]))
    clip = concatenate_videoclips([VideoFileClip(video) for video in videos])
    clip.write_videofile(f"{video_dir}.mp4")
    shutil.rmtree(video_dir)


def set_seed_everywhere(env: gym.Env, seed=0):
    """
    Set seed for all randomness sources
    """
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_space_shape(space):
    """
    Return the shape of the gym.Space object
    """
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Box):
        if len(space.shape) == 1:
            return space.shape[0]
        else:
            return space.shape
    else:
        raise ValueError(f"Space not supported: {space}")
