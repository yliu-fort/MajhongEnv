import random

class RandomDiscardAgent:
    """
    A dummy agent that discards a random tile from the current hand.
    Supports fixed seed for reproducibility.
    """
    def __init__(self, env, backbone=None, seed=None):
        self.env = env
        self.random_generator = random.Random(seed)  # Use a separate Random instance with optional seed

    def save_model(self, path=""):
        pass

    def load_model(self, path=""):
        pass

    def predict(self, _):
        """
        Given the current observation (list of tiles in hand),
        return a random tile to discard from the hand.
        """
        if self.env is None or not hasattr(self.env, "action_masks"):
            raise ValueError("RandomDiscardAgent requires an environment with an action_masks method")

        action_masks = self.env.action_masks()
        valid_action_list = [i for i in list(range(len(action_masks))) if action_masks[i] == 1]

        # Randomly pick one tile from the hand to discard
        if valid_action_list:
            return self.random_generator.choice(valid_action_list)

        return 252