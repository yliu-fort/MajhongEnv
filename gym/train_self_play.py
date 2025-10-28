import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "gym"))
import time
import numpy as np
from functools import partial
from mahjong_gym import MahjongEnvPettingZoo
from agent.random_discard_agent import RandomDiscardAgent
from agent.rule_based_agent import RuleBasedAgent
from my_types import Response, Seat
from concurrent.futures import ProcessPoolExecutor


def evaluate_model(episodes=10, start=0, step=1):
    # 创建环境
    env = MahjongEnvPettingZoo(num_players=4)
    agent = RuleBasedAgent(env, backbone="resnet50")
    
    # 加载训练好的模型
    agent.load_model("legacy/model_weights/latest.pt")
 
    total_dscores = np.zeros(4, dtype=np.int32)
    for ep in range(start, episodes, step):
        obs, _ = env.reset()
        done = False
        while not done:
            # 用智能体来选动作
            actions = {_: agent.predict(obs[_]["observation"]) for _ in range(env.num_players)}
            obs, rewards, terminations, _, _ = env.step(actions)
            done = any(terminations.values())

        total_dscores += np.array(env.info["scores"]) - 250
        print(f"Episode {ep} - 分数板：{total_dscores}", env.info["scores"])
        print(env.info["msg"])
        #with open(f'../log_analyser/paipu/evaluate_log_{ep}.mjlog', "w") as f:
        #    f.write(info["log"])


if __name__ == "__main__":
    # Serial
    start = time.monotonic()
    evaluate_model(episodes=10)
    now = time.monotonic()
    print(f"[S] Elapsed time = {(now - start)/ 10.0} s")