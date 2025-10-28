import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
from mahjong_env import MahjongEnvPettingZoo
import time
from pettingzoo.test import parallel_api_test


if __name__ == "__main__":
    # Serial
    start = time.monotonic()
    env = MahjongEnvPettingZoo(num_players=4)
    parallel_api_test(env, num_cycles=10000)
    now = time.monotonic()
    print(f"[S] Elapsed time = {(now - start)/ 10.0} s")
