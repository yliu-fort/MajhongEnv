"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy

from mahjong_gym import MahjongEnvPettingZoo
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class VecActionMaskAdapter(VecEnvWrapper):
    """
    从 VecEnv 的 obs / infos 中提取每个子环境的 action_mask，
    并对外提供 action_masks()，供 MaskablePPO 调用。
    """
    def __init__(self, venv):
        super().__init__(venv, venv.observation_space, venv.action_space)
        self._last_masks = None  # shape: (n_envs, action_dim)

    def reset(self, **kwargs):
        out = self.venv.reset(**kwargs)
        # 兼容 Gymnasium / SB3 v2 的返回 (obs, info)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            self._last_masks = self._extract_masks(obs, info)
            return obs, info
        # 旧接口（极少见）
        obs = out
        self._last_masks = self._extract_masks(obs, None)
        return obs

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        res = self.venv.step_wait()
        # 兼容 SB3 v2（Gymnasium）5元组 & 旧 4元组
        if len(res) == 5:
            obs, rewards, terms, truncs, infos = res
            self._last_masks = self._extract_masks(obs, infos)
            return obs, rewards, terms, truncs, infos
        else:
            obs, rewards, dones, infos = res
            self._last_masks = self._extract_masks(obs, infos)
            return obs, rewards, dones, infos

    # 供 sb3-contrib 读取的入口
    def action_masks(self):
        return self._last_masks

    def _extract_masks(self, obs, infos):
        # 优先从观测 Dict 提取（VecEnv 下通常是 batched 的）
        if isinstance(obs, dict) and "action_mask" in obs:
            return np.asarray(obs["action_mask"])
        # 其次尝试从 infos（VecEnv 通常是 list[dict]）提取
        if isinstance(infos, (list, tuple)):
            masks = [info.get("action_mask") for info in infos]
            if all(m is not None for m in masks):
                return np.asarray(masks)
        if isinstance(infos, dict) and "action_mask" in infos:
            return np.asarray(infos["action_mask"])
        return None


# 关键：把 Supersuit 的 VecEnv 变成 SB3 认可的 VecEnv
class SB3Compat(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv, venv.observation_space, venv.action_space)
    
    def reset(self):
        return super().reset()
    
    def step_wait(self):
        return super().step_wait()


def train_mjai(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # ② （强烈建议）用 SuperSuit 包装，确保满足转换假设
    #    - pad_observations/pad_action_space：补齐不同代理之间的维度差异
    #    - black_death_v3：当某个代理“死亡/无动作”时，保持其存在，输出零观测+空动作
    #env = ss.pad_observations_v0(env)
    #env = ss.pad_action_space_v0(env)
    #env = ss.black_death_v3(env)

    # ③ ParallelEnv -> SB3 VecEnv
    print("has action_masks:", hasattr(env, "action_masks"))
    
    env = ss.pettingzoo_env_to_vec_env_v1(env)  # 每个“子环境”对应一个agent
    env = VecActionMaskAdapter(env)  # TODO: Fix this issue: TypeError: cannot pickle '_abc._abc_data' object Exception ignored in: <function VectorEnv.__del__ at 0x1088cd6c0>
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=0, base_class="stable_baselines3")
    

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=4,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


if __name__ == "__main__":
    env_fn = MahjongEnvPettingZoo
    env_kwargs = {}

    # Train a model (takes ~3 minutes on GPU)
    train_mjai(env_fn, steps=196_608, seed=0, **env_kwargs)