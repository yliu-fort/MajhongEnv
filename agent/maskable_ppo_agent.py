import gym
import torch
import torch.nn as nn
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy, MaskableActorCriticCnnPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    
class MaskablePPOAgentV2:
    def __init__(self, env: gym.Env):
        self.env = env
        # 注意，这里用的是 MaskablePPO，而不是原来的 PPO
        # 策略也需要选择 MaskableActorCriticPolicy 或者 MaskableMultiInputPolicy
        self.model = MaskablePPO(
            policy=MaskableActorCriticCnnPolicy,
            env=env,
            verbose=1,
            policy_kwargs=dict(
            features_extractor_class=ResNet34FeatureExtractor,
            features_extractor_kwargs=dict(), \
            net_arch=dict(pi=[256, ], vf=[256, ]),  # encoder-only architecture
            ),
            tensorboard_log="./tensorboard_logs/"
        )

    def train(self, total_timesteps=100000):
        self.model.learn(total_timesteps=total_timesteps)

    def save_model(self, path="mppov2_mahjong"):
        self.model.save(path)

    def load_model(self, path="mppov2_mahjong"):
        self.model = MaskablePPO.load(path, env=self.env)

    def predict(self, observation):
        # 与普通 PPO 用法类似，但注意要传入的 observation 必须与训练时对应
        # 如果环境 observation 是个字典，这里也要按字典格式传入
        # 在 sb3-contrib 中，model.predict 会自动根据 observation["action_mask"] 屏蔽无效动作
        action_masks = get_action_masks(self.env)
        action, _states = self.model.predict(observation, deterministic=True, action_masks=action_masks)
        return action