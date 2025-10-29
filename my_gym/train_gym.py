"""Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Author: Elliot (https://github.com/elliottower)
"""

import glob
import os
import time

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnvWrapper
from sb3_contrib.common.wrappers import ActionMasker
import torch.nn as nn
from gymnasium import spaces
from mahjong_gym import MahjongEnvGym
from res_1d_extractor import ResNet1DExtractor

def mask_fn(env):
    # Custom logic to determine valid actions
    return env.action_mask()  # Returns boolean array (True=valid)


def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ActionMasker(env, mask_fn)
    
    env.reset()  # Must call reset() in order to re-define the spaces

    model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, verbose=1)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


if __name__ == "__main__":
    env_fn = MahjongEnvGym

    env_kwargs = {}

    # Evaluation/training hyperparameter notes:
    # 10k steps: Winrate:  0.76, loss order of 1e-03
    # 20k steps: Winrate:  0.86, loss order of 1e-04
    # 40k steps: Winrate:  0.86, loss order of 7e-06

    # Train a model against itself (takes ~20 seconds on a laptop CPU)
    train_action_mask(env_fn, steps=20_480, seed=0, **env_kwargs)