import glob
import os, sys
from sb3_contrib import MaskablePPO


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
        print(f"Load Policy from {latest_policy}.")

    def predict(self, obs):
        return int(
                    self._model.predict(
                        obs, action_masks=obs["action_mask"], deterministic=True
                    )[0]
                )
