"""Uses Stable-Baselines3 to train agents in the MJAI environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Author: Y. Liu (https://github.com/yliu-fort)
"""

import glob
import os, sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
import gymnasium as gym
import numpy as np
import psutil
from functools import partial
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from mahjong_gym import MahjongEnvGym
from res_1d_extractor import ResNet1DExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from ppo_agent import MaskablePPOAgent, MaskablePPOAgentPool

IMITATION_REWARD = False
RIICHI_REWARD = False
AGARI_REWARD = False
SCORE_DELTA_REWARD = True
RANK_BONUS_REWARD = True
FURITEN_PENALTY = False


# --------- 1) 你的环境 & 掩码函数（必须是顶层可 pickl e 的函数！）---------
def make_single_env(env_fn, rank: int, seed: int = 0):
    def _init():
        env = env_fn()
        print("spawn env in PID:", os.getpid())
        print(f'Imitation Reward: {"on" if env._imitation_reward else "off"}')
        env.reset()
        return env
    return _init


def train_mjai(env_fn, steps=10_000, seed=0, continue_training=False, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    print(f"物理核心数: {psutil.cpu_count(logical=False)}, 逻辑核心数: {psutil.cpu_count(logical=True)}")
    cpu_count = psutil.cpu_count(logical=False) or 1
    max_workers = min(24, max(1, cpu_count - 1 if cpu_count > 1 else 1))
    
    env = env_fn()
    n_envs = max_workers
    
    # Windows/macOS 推荐 spawn（SB3 内部会处理）；确保在 __main__ 保护下运行
    env_fns = [make_single_env(env_fn, i, seed) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)          # 多进程
    vec_env = VecMonitor(vec_env)             # 记录回报/长度等指标

    # 可选：限制 PyTorch 线程数，避免与多进程争抢
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    
    print(f"Starting training on {str(env.metadata['name'])}.")

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
        vec_env,
        verbose=2,
        learning_rate=3e-4,
        batch_size=128,
        n_steps=4096 // n_envs,           # 更小
        n_epochs=3,            # 减少优化开销
        target_kl=0.1,
        gae_lambda=0.95, gamma=0.993,
        policy_kwargs=policy_kwargs,
        device = "cuda" if torch.cuda.is_available else "cpu",
        tensorboard_log=f'runs/{time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}'
    )

    if continue_training:
        try:
            latest_policy = max(
                glob.glob(f"model_weights/{env.metadata['name']}*.zip"), key=os.path.getctime
            )
            print(f"Load from latest policy {latest_policy}.")
            model = MaskablePPO.load(latest_policy, vec_env)
        except ValueError:
            print("Policy not found, start from scratch...")
            
    model.set_random_seed(seed)
    save_freq = 1048576
    save_freq = max(save_freq // n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="model_weights",
        name_prefix=env.unwrapped.metadata.get('name', 'model'),
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=2
    )
    
    '''
    try:
        prefix = env.unwrapped.metadata.get('name', 'model')
        ckpt_pattern = os.path.join("model_weights", f"{prefix}*.zip")
        ckpts = glob.glob(ckpt_pattern)
        if len(ckpts) > 10:
            ckpts.sort(key=os.path.getctime, reverse=True)
            for old in ckpts[10:]:
                try:
                    os.remove(old)
                except OSError:
                    pass
    except Exception:
        pass
    '''


    model.learn(total_timesteps=steps, callback=checkpoint_callback, progress_bar=True, reset_num_timesteps=False)
    
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
            glob.glob(f"model_weights/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPOAgent(env)

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
                act = model.predict(obs)
            obs, reward, termination, truncation, info = env.step(act)
    env.close()

    # Avoid dividing by zero
    winrate = score / num_games
    avg_dan = dan / num_games
    
    print("Winrate: ", winrate)
    print("Avg. Dan: ", avg_dan + 1)


if __name__ == "__main__":
    env_fn = partial(MahjongEnvGym, imitation_reward=IMITATION_REWARD, opponent_fn=MaskablePPOAgent)
    env_kwargs = {}

    # Train a model against itself (takes ~20 seconds on a laptop CPU)
    train_mjai(env_fn, steps=20_480_000, seed=42, **env_kwargs)
    
    eval_mjai(env_fn)
