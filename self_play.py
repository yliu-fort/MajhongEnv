import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))

import gymnasium as gym
import numpy as np
from mahjong_env import MahjongEnv
from agent.random_discard_agent import RandomDiscardAgent
from agent.visual_agent import VisualAgent

def evaluate_model(episodes=10):
    # 创建环境
    env = MahjongEnv(num_players=4)
    agent = VisualAgent(env, backbone="resnet18")
    
    # 加载训练好的模型
    agent.load_model("model_weights/resnet18.pt")
 
    total_dscores = np.zeros(4, dtype=np.int32)
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            # 用智能体来选动作
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)

        total_dscores += np.array(info["scores"]) - 250
        print(f"Episode {ep} - 分数板：{total_dscores}", info["scores"])
        print(info["msg"])
        with open(f'./output/paipu/evaluate_log_{ep}.mjlog', "w") as f:
            f.write(info["log"])
 

def evaluate_model_multi(model_paths=[], model_classes=[], episodes=100, num_players=4):
    # 创建环境
    env = MahjongEnv(num_players=num_players)
    agents = []
    
    # 加载训练好的模型
    assert len(model_paths) == num_players
    for model_path, model_class in zip(model_paths, model_classes):
        agent = model_class(env)
        agent.load_model("model_weights/"+model_path)
        agents.append(agent)
        del agent

    total_dscores = np.zeros(4, dtype=np.int32)
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            # 用智能体来选动作
            player = env.current_player
            action = agents[player].predict(obs)
            obs, _, done, info = env.step(action)

        total_dscores += np.array(info["scores"]) - 250
        print(f"Episode {ep} - 分数板：{total_dscores}", info["scores"])
        print(info["msg"])

if __name__ == "__main__":
    evaluate_model()
    #model_files = [f for f in os.listdir("model_weights/") if f.startswith("resnet_")]
    #evaluate_model_multi(["","","",model_files[-1]],
    #                     [RandomDiscardAgent, RandomDiscardAgent, RandomDiscardAgent, MaskablePPOAgent])