import os
import sys
from typing import Any, List

import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import gymnasium as gym
import myosuite

import DynSyn
from mani_wrapper import ManiWrapper

def create_env(env_name: str, single_env_kwargs: dict, wrapper_list: dict):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    env = gym.make(env_name, **single_env_kwargs)

    for wrapper_name, wrapper_args in wrapper_list.items():
        try:
            env = eval(wrapper_name)(env, **wrapper_args)
        except NameError:
            print(f"Wrapper {wrapper_name} not found!")
            raise NameError

    return env

class ManiAgent:
    def __init__(self, checkpoint_path, env_name, single_env_kwargs, wrapper_list) -> None:
        dummy_env = create_env(env_name, single_env_kwargs, wrapper_list)
        ckp_path = checkpoint_path
        self.vec_norm = VecNormalize.load(os.path.join(ckp_path, "best_env.zip"), DummyVecEnv([lambda: dummy_env]))
        self.model = DynSyn.SAC_DynSyn.load(os.path.join(ckp_path, "best_model.zip"), env=self.vec_norm)

    def __call__(self, obs):
        obs = self.vec_norm.normalize_obs(obs)
        action, _states = self.model.predict(obs)
        return action

def load_mani(checkpoint_path, obs_keys):
    return ManiAgent(
        checkpoint_path=checkpoint_path,
        env_name="myoChallengeBimanual-v0",
        single_env_kwargs={
            "obs_keys": obs_keys
        },
        wrapper_list={
            "ManiWrapper": {}
        }
    )

