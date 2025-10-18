import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "agent"))

import numpy as np
from mahjong_env import MahjongEnv
from agent.random_discard_agent import RandomDiscardAgent
from agent.rule_based_agent import RuleBasedAgent
from agent.visual_agent import VisualAgent

def evaluate_model(episodes=100):
    # 创建环境
    env = MahjongEnv(num_players=4)
    agent = VisualAgent(env, backbone="resnet50")
    
    # 加载训练好的模型
    agent.load_model("model_weights/latest.pt")
 
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
        with open(f'../log_analyser/paipu/evaluate_log_{ep}.mjlog', "w") as f:
            f.write(info["log"])
 

def evaluate_model_multi(model_paths=[], models=[], model_classes=[], episodes=500, num_players=4):
    # 创建环境
    env = MahjongEnv(num_players=num_players, num_rounds=8)
    agents = []
    
    # 加载训练好的模型
    assert len(models) == num_players
    for i, (model, model_class) in enumerate(zip(models, model_classes)):
        agent = model_class(env, backbone=model)
        agent.load_model(model_paths[i])
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
        with open(f'../log_analyser/paipu/evaluate_log_{ep}.mjlog', "w") as f:
            f.write(info["log"])

if __name__ == "__main__":
    evaluate_model()
    '''
    model_files = ["model_weights/step_20000.pt", 
                   "model_weights/step_25000.pt",
                   "model_weights/step_30000.pt", 
                   "model_weights/step_15000.pt"]
    evaluate_model_multi(model_files, ["resnet50","resnet50","resnet50","resnet50"],
                     [VisualAgent, VisualAgent, VisualAgent, VisualAgent])
    '''