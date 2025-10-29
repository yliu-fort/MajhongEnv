"""Uses Stable-Baselines3 to train agents in the Connect Four environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Author: Elliot (https://github.com/elliottower)
"""

import glob
import os, sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
import gymnasium as gym
import numpy as np
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnvWrapper
from sb3_contrib.common.wrappers import ActionMasker
from mahjong_gym import MahjongEnvGym
from res_1d_extractor import ResNet1DExtractor
from agent.rule_based_agent import RuleBasedAgent

def mask_fn(env):
    # Custom logic to determine valid actions
    return env.action_mask()  # Returns boolean array (True=valid)


def train_mjai(env_fn, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn(**env_kwargs)
    env = ActionMasker(env, mask_fn)
    
    print(f"Starting training on {str(env.metadata['name'])}.")
    
    env.reset()  # Must call reset() in order to re-define the spaces

    # Model
    policy_kwargs = dict(
        features_extractor_class=ResNet1DExtractor,
        features_extractor_kwargs=dict(),
        share_features_extractor=True,   # 默认就是 True，这里强调一下
        net_arch=dict(pi=[], vf=[64,]),             # MLP
        activation_fn=nn.SiLU,           # 与 SB3 默认一致，可改 ReLU
    )

    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        verbose=2,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=1024,           # 更小
        n_epochs=10,            # 减少优化开销
        gae_lambda=0.95, gamma=0.993,
        policy_kwargs=policy_kwargs
    )
    #model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, verbose=1)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_mjai(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode, **env_kwargs)

    print(
        f"Starting evaluation."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)
    alt = RuleBasedAgent(env)

    score = 0
    dan = 0
    
    for i in range(num_games):
        obs, _ = env.reset(seed=i)

        while True:
            if env.done:
                print(env.info["msg"])
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env._focus_player == env.info["rank"][0]
                ):
                    score += 1
                    dan += env.info["rank"].index(env._focus_player)
                break
            else:
                act = int(
                    model.predict(
                        obs, action_masks=obs["action_mask"], deterministic=True
                    )[0]
                )
                #act = alt.predict(env.get_observation(env._focus_player))
            obs, reward, termination, truncation, info = env.step(act)
    env.close()

    # Avoid dividing by zero
    winrate = score / num_games
    avg_dan = dan / num_games
    
    print("Winrate: ", winrate)
    print("Avg. Dan: ", avg_dan + 1)



if __name__ == "__main__":
    env_fn = MahjongEnvGym
    env_kwargs = {}

    # Train a model against itself (takes ~20 seconds on a laptop CPU)
    train_mjai(env_fn, steps=20_480_0, seed=42, **env_kwargs)
    
    eval_mjai(env_fn)