import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
 
class PPOAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.model = PPO(
            policy=MultiInputPolicy,
            env=env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
 
    def train(self, total_timesteps=100000):
        self.model.learn(total_timesteps=total_timesteps)
 
    def save_model(self, path="ppo_mahjong"):
        self.model.save(path)
 
    def load_model(self, path="ppo_mahjong"):
        self.model = PPO.load(path, env=self.env)
 
    def predict(self, observation):
        # 如果当前状态是和牌状态，直接返回和牌动作
        if self.env and (self.env.phase == "tsumo" or self.env.phase == "ron"):
            return 140
        
        # 推理时获取动作
        action, _states = self.model.predict(observation, deterministic=True)
        return action