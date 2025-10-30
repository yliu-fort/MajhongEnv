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
        self._selected = 0

    def rselect(self):
        if len(self._pool) == 0: raise ValueError("Pool is empty!!")
        self._selected = random.choice(list(range(len(self._pool))))
        
    def predict(self, obs):
        return self._pool[self._selected].predict(obs)