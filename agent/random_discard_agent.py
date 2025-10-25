import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import gymnasium as gym
import random
from my_types import Response, ActionSketch, Seat, ActionType
from mahjong_features import get_action_type_from_index, NUM_ACTIONS

'''
class Response:
    room_id: str
    step_id: int
    request_id: str
    from_seat: Seat
    chosen: ActionSketch
'''
class RandomDiscardAgent:
    """
    A dummy agent that discards a random tile from the current hand.
    Supports fixed seed for reproducibility.
    """
    def __init__(self, env: gym.Env, backbone=None, seed=None):
        self.env = env
        self.random_generator = random.Random(seed)  # Use a separate Random instance with optional seed

    def save_model(self, path=""):
        pass

    def load_model(self, path=""):
        pass

    def predict(self, obs):
        """
        Given the current observation (list of tiles in hand),
        return a random tile to discard from the hand.
        """
        if self.env is None or not hasattr(self.env, "action_masks"):
            raise ValueError("RandomDiscardAgent requires an environment with an action_masks method")

        #who = obs[1]["who"]
        #action_masks = self.env.action_masks(who)
        action_masks = obs.legal_actions_mask
        valid_action_list = [i for i in list(range(NUM_ACTIONS)) if action_masks[i]]
        if sum(action_masks) == 0:
            return None
        elif sum(action_masks) == 1:
            action_id = valid_action_list[0]
            return ActionSketch(action_type=get_action_type_from_index(action_id), payload={"action_id": action_id})
        
        # Randomly pick one tile from the hand to discard
        if valid_action_list:
            action_id = self.random_generator.choice(valid_action_list)
            return ActionSketch(action_type=get_action_type_from_index(action_id), payload={"action_id": action_id})

        return ActionSketch(action_type=ActionType.PASS, payload={"action_id": 252})
