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


class MaskablePPOAgentPool:
    _pool: list = []
    
    @staticmethod
    def register(prefix=""):
        MaskablePPOAgentPool._pool.clear()
        try:
            ckpt_pattern = os.path.join("model_weights", f"{prefix}*.zip")
            ckpts = glob.glob(ckpt_pattern)
            ckpts.sort(key=os.path.getctime, reverse=True)
            for c in ckpts[:10]:
                try:
                    MaskablePPOAgentPool._pool.append(MaskablePPO.load(c))
                    print(f"Load Frozen Policy from {c}.")
                except OSError:
                    pass
        except Exception:
            pass
    
    def __init__(self, env):
        self._selected = 0
        self._strong_rule_based = RuleBasedAgent(env)

    def rselect(self):
        if random.random() < 0.1 or len(MaskablePPOAgentPool._pool) == 0:
            self._selected = -1
        else:
            self._selected = random.choice(list(range(len(MaskablePPOAgentPool._pool))))
        
    def predict(self, obs):
        if self._selected == -1:
            _op = self._strong_rule_based
        else:
            _op = MaskablePPOAgentPool._pool[self._selected]
        return _op.predict(obs)