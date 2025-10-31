import glob
import os, sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
from sb3_contrib import MaskablePPO
from rule_based_agent import RuleBasedAgent


class MaskablePPOAgent:
    def __init__(self, env):
        try:
            latest_policy = max(
                glob.glob(f"model_weights/{env.metadata['name']}*.zip"), key=os.path.getctime
            )
        except ValueError:
            print("Policy not found.")
            exit(0)

        self._model = MaskablePPO.load(latest_policy)
        #self._model.set_training_mode(False) # For newer version SB3
        self._model.policy.eval()
        print(f"Load Frozen Policy from {latest_policy}.")

    def predict(self, obs):
        return int(
                    self._model.predict(
                        obs, action_masks=obs["action_mask"], deterministic=True
                    )[0]
                )

    def __getitem__(self, _):
        return self

class MaskablePPOAgentWrapper:
    def __init__(self, model):
        self._model = model
        #self._model.set_training_mode(False) # For newer version SB3
        self._model.policy.eval()

    def predict(self, obs):
        return int(
                    self._model.predict(
                        obs, action_masks=obs["action_mask"], deterministic=True
                    )[0]
                )

class MaskablePPOAgentPool:
    def __init__(self, env):
        self._pool=[RuleBasedAgent(env),]
        try:
            ckpt_pattern = os.path.join("model_weights", f"{env.metadata.get('name')}*.zip")
            ckpts = glob.glob(ckpt_pattern)
            ckpts.sort(key=os.path.getctime, reverse=True)
            for c in ckpts[:10]:
                try:
                    self._pool.append(MaskablePPOAgentWrapper(MaskablePPO.load(c, device='cpu')))
                    print(f"Pool loads Frozen Policy from {c}.")
                except OSError:
                    pass
        except Exception:
            pass
        self._selected = [0, 0, 0, 0]

    def shuffle(self):
        if len(self._pool) == 0: raise ValueError("Pool is empty!!")
        for idx in range(4):
            self._selected[idx] = random.choice(list(range(len(self._pool))))
    
    def __getitem__(self, index):
        return self._pool[self._selected[index%4]]
    

# 2) Callback：按 explained_variance 自适应调整 reward scale
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
import numpy as np

class ImitationWeightProxyWrapper(gym.Wrapper):
    """
    仅作为“控制通道”，通过 env_method 在 VecEnv 中统一修改底层环境的
    `unwrapped._imitation_reward_weight` 属性。不会改 reward。
    """
    def __init__(self, env, attr_name: str = "_imitation_reward_weight",
                 init_weight: float = 0.0, min_weight: float = 0.001, max_weight: float = 1.0,
                 clip_weight: bool = True):
        super().__init__(env)
        self._attr = str(attr_name)
        self._min = float(min_weight)
        self._max = float(max_weight)
        self._clip = bool(clip_weight)

        # 若提供 init_weight，则在构造时设入底层 env
        self.set_imitation_weight(init_weight)

    # ------- 提供给 env_method 的 API -------
    def set_imitation_weight(self, new_weight: float):
        if self._clip:
            new_weight = float(np.clip(new_weight, self._min, self._max))
        env_u = getattr(self.env, "unwrapped", self.env)
        setattr(env_u, self._attr, float(new_weight))
        return float(new_weight)  # env_method 会返回到调用方（可选）

    def get_imitation_weight(self) -> float:
        env_u = getattr(self.env, "unwrapped", self.env)
        return float(getattr(env_u, self._attr))

    # 便捷增减
    def mul_imitation_weight(self, factor: float):
        cur = self.get_imitation_weight()
        return self.set_imitation_weight(cur * float(factor))

    def add_imitation_weight(self, delta: float):
        cur = self.get_imitation_weight()
        return self.set_imitation_weight(cur + float(delta))


class EVRewardScaleCallback(BaseCallback):
    def __init__(
        self,
        target_ev: float = 0.2,   # 目标解释度
        tol: float = 0.1,         # 容忍带
        up: float = 1.05,         # EV 太低时微增倍率
        down: float = 0.90,       # EV 太高时下调倍率
        min_scale: float = 0.001,
        max_scale: float = 1.0,
        every_n_rollouts: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_ev = target_ev
        self.tol = tol
        self.up = up
        self.down = down
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.every = max(1, int(every_n_rollouts))
        self._k = 0

    def _on_step(self) -> bool:
        # 必须实现；不做事就返回 True
        return True
    
    def _on_rollout_end(self) -> bool:
        self._k += 1
        if self._k % self.every != 0:
            return True

        buf = self.model.rollout_buffer
        values = buf.values.copy().flatten()
        returns = buf.returns.copy().flatten()

        ev = explained_variance(values, returns)
        if np.isnan(ev):
            return True

        # 计算倍率更新因子
        if ev < self.target_ev - self.tol:
            factor = self.up
        elif ev > self.target_ev + self.tol:
            factor = self.down
        else:
            factor = 1.0

        # 当前并行 env 的权重（求平均供日志参考）
        ws = self.training_env.env_method("get_imitation_weight")
        cur = float(np.mean(ws)) if len(ws) else 1.0
        new_w_list = self.training_env.env_method("mul_imitation_weight", factor)

        new_mean = float(np.mean(new_w_list)) if len(new_w_list) else cur
        self.logger.record("adaptive/ev_rollout", float(ev))
        self.logger.record("adaptive/imit_weight", new_mean)
        self.logger.record("adaptive/imit_factor", float(factor))
        
        if self.verbose:
            print(f"[EV→_imit_w] EV={ev:.3f} | w_mean {cur:.4f} -> {new_mean:.4f} (x{factor:.3f})")
        return True
