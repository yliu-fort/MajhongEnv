import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "agent"))

import gymnasium as gym
import numpy as np
from mahjong_env import MahjongEnv
from agent.random_discard_agent import RandomDiscardAgent
from agent.rule_based_agent import RuleBasedAgent
from my_types import Response, Seat
import time

def evaluate_model(episodes=100):
    # 创建环境
    env = MahjongEnv(num_players=4)
    agent = RandomDiscardAgent(env, backbone="resnet50")
    
    # 加载训练好的模型
    agent.load_model("legacy/model_weights/latest.pt")
 
    total_dscores = np.zeros(4, dtype=np.int32)
    for ep in range(episodes):
        obs = env.reset()
        done = False
        step_counter = 0
        while not done:
            # 用智能体来选动作
            actions = {
                seat: Response(
                    room_id="",
                    step_id=step_counter,
                    request_id=f"req-{step_counter}-{seat}",
                    from_seat=Seat(seat),
                    chosen=agent.predict(obs[seat]),
                )
                for seat in range(env.num_players)
            }
            obs, rewards, terminations, truncations, info = env.step(actions)
            step_counter += 1

            done = False
            if isinstance(terminations, dict):
                done = any(bool(flag) for flag in terminations.values())
            else:
                done = bool(terminations)
            if not done:
                if isinstance(truncations, dict):
                    done = any(bool(flag) for flag in truncations.values())
                else:
                    done = bool(truncations)

        total_dscores += np.array(info["scores"]) - 250
        print(f"Episode {ep} - 分数板：{total_dscores}", info["scores"])
        print(info["msg"])
        #with open(f'../log_analyser/paipu/evaluate_log_{ep}.mjlog', "w") as f:
        #    f.write(info["log"])


if __name__ == "__main__":
    start = time.monotonic_ns() // 1_000_000
    evaluate_model()
    now = time.monotonic_ns() // 1_000_000
    print(f"Elapsed time = {(now - start) / 1000.0 / 100.0} s")