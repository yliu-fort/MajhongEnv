"""Run the Mahjong environment with the Kivy GUI wrapper."""

import os
import sys

ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(os.path.join(ROOT, "agent"))

from agent.random_discard_agent import RandomDiscardAgent
from mahjong_env import MahjongEnv
from mahjong_wrapper_kivy import MahjongEnvKivyWrapper


def main(episodes: int = 1) -> None:
    env_core = MahjongEnv(num_players=4)
    env = MahjongEnvKivyWrapper(env=env_core)
    agent = RandomDiscardAgent(env_core)

    try:
        for _ in range(episodes):
            observation = env.reset()
            done = False
            while not done:
                action = agent.predict(observation)
                observation, _reward, done, _info = env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
