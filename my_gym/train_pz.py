"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time
import numpy as np
import supersuit as ss
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnvWrapper

import torch.nn as nn
from gymnasium import spaces
from mahjong_gym import MahjongEnvPettingZoo
from res_1d_extractor import ResNet1DExtractor

class SB3MaskVecAdapter(VecEnvWrapper):
    """
    适配 Supersuit ConcatVecEnv 到 SB3：
    - reset(): 只返回 obs（丢弃 info，SB3 需要这样）
    - step_wait(): 如果是 (obs, rew, term, trunc, infos) 则合并成 dones=term|trunc
    - action_masks(): 从 obs/infos 或底层 vec_envs 聚合出掩码
    """
    def __init__(self, venv):
        super().__init__(venv, venv.observation_space, venv.action_space)
        self._last_masks = None

    def reset(self, **kwargs):
        out = self.venv.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, None
        self._last_masks = self._extract_masks(obs, info)
        return obs  # 关键：SB3 期望只返回 obs

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        res = self.venv.step_wait()
        if len(res) == 5:
            obs, rewards, terminations, truncations, infos = res
            dones = np.logical_or(terminations, truncations)
            self._last_masks = self._extract_masks(obs, infos)
            return obs, rewards, dones, infos
        else:
            # 旧式 4 元组
            obs, rewards, dones, infos = res
            self._last_masks = self._extract_masks(obs, infos)
            return obs, rewards, dones, infos

    def action_masks(self):
        return self._last_masks

    def _extract_masks(self, obs, infos):
        # 1) 从 batched 观测字典里取（推荐做法）
        if isinstance(obs, dict) and "action_mask" in obs:
            return np.asarray(obs["action_mask"])

        # 2) 从 infos 里聚合（VecEnv: list[dict] 或 dict）
        if isinstance(infos, (list, tuple)):
            masks = [i.get("action_mask") for i in infos]
            if masks and all(m is not None for m in masks):
                return np.asarray(masks)
        if isinstance(infos, dict) and "action_mask" in infos:
            return np.asarray(infos["action_mask"])

        # 3) 兜底：如果底层有 vec_envs 且它们各自实现了 action_masks()，把它们拼起来
        if hasattr(self.venv, "vec_envs"):
            try:
                parts = []
                for v in self.venv.vec_envs:
                    if hasattr(v, "action_masks"):
                        m = v.action_masks()
                        if m is not None:
                            parts.append(m)
                if parts:
                    return np.concatenate(parts, axis=0)
            except Exception:
                pass

        return None

    # 覆盖这三个，确保 MaskablePPO 的探测/调用不会掉到内层 ConcatVecEnv
    def has_attr(self, attr_name):
        if attr_name == "action_masks":
            return True
        inner = getattr(self.venv, "has_attr", None)
        if callable(inner):
            try:
                return inner(attr_name)
            except Exception:
                pass
        return hasattr(self.venv, attr_name)

    def get_attr(self, attr_name, indices=None):
        if attr_name == "action_masks":
            m = self.action_masks()
            return [row for row in np.asarray(m)]
        inner = getattr(self.venv, "get_attr", None)
        if callable(inner):
            return inner(attr_name, indices=indices)
        raise AttributeError(f"{type(self.venv).__name__} has no get_attr")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name == "action_masks":
            m = self.action_masks()
            return [row for row in np.asarray(m)]
        inner = getattr(self.venv, "env_method", None)
        if callable(inner):
            return inner(method_name, *args, indices=indices, **kwargs)
        raise AttributeError(f"{type(self.venv).__name__} has no env_method")




def train_mjai(
    env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # ③ ParallelEnv -> SB3 VecEnv
    env = ss.pettingzoo_env_to_vec_env_v1(env)  # 每个“子环境”对应一个agent
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=0, base_class="stable_baselines3")
    env = SB3MaskVecAdapter(env)
    

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
        batch_size=16,
        n_steps=256,           # 更小
        n_epochs=10,            # 减少优化开销
        gae_lambda=0.95, gamma=0.993,
        policy_kwargs=policy_kwargs
    )
    print(model.policy)  # 会打印出 features extractor 和 net_arch，确认是否共享

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()



def eval_mjai(env_fn, episodes=10, start=0, step=1):
    # Evaluate a trained agent vs a random agent
    env = env_fn(**env_kwargs)

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

    agent = MaskablePPO.load(latest_policy)
 
    total_dscores = np.zeros(4, dtype=np.int32)
    for ep in range(start, episodes, step):
        obs, _ = env.reset()

        while env.agents:
            # 用智能体来选动作
            actions = {}
            for i in range(env.num_players):
                observation, action_mask = obs[i].values()
                actions[i] = agent.predict(obs[i], action_masks=action_mask, deterministic=True)[0]
                
            obs, _, _, _, _ = env.step(actions)

        total_dscores += np.array(env.info["scores"]) - 250
        print(f"Episode {ep} - 分数板：{total_dscores}", env.info["scores"])
        print(env.info["msg"])
        #with open(f'../log_analyser/paipu/evaluate_log_{ep}.mjlog', "w") as f:
        #    f.write(info["log"])


if __name__ == "__main__":
    env_fn = MahjongEnvPettingZoo
    env_kwargs = {}

    # Train a model (takes ~3 minutes on GPU)
    train_mjai(env_fn, steps=196_608_0, seed=0, **env_kwargs)
    
    eval_mjai(env_fn, **env_kwargs)
