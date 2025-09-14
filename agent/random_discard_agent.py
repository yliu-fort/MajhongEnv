import gymnasium as gym
import random

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

    def predict(self, _):
        """
        Given the current observation (list of tiles in hand),
        return a random tile to discard from the hand.
        """
        if self.env is None or not hasattr(self.env, "action_masks"):
            raise ValueError("RandomDiscardAgent requires an environment with an action_masks method")

        action_masks = self.env.action_masks()
        valid_action_list = [i for i in list(range(14)) if action_masks[i] == 1]
        valid_confirm_list = action_masks[-1]

        # Randomly pick one tile from the hand to discard
        if valid_action_list:
            action = self.random_generator.choice(valid_action_list)
        else:
            action = 0
        if valid_confirm_list:
            confirm = self.random_generator.choice([0, 1])
        else:
            confirm = 0

        return (action, confirm)
    

if __name__=="__main__":
    # Test the RandomDiscardAgent with a dummy environment

    class DummyEnv:
        def action_masks(self):
            return [1] * 14 + [0, 0]

    agent = RandomDiscardAgent(DummyEnv(), seed=77777)
    observation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    print("Test observation:", observation)
    action = agent.predict({"hands": observation})
    print("Randomly selected action:", action)

    observation = [-1, 135, -1, -1, -1]
    print("Test observation:", observation)
    action = agent.predict({"hands": observation})
    print("Randomly selected action:", action)

    observation = [-1, -1, -1, -1, -1]
    print("Test observation:", observation)
    try:
        action = agent.predict({"hands": observation})
    except ValueError as e:
        print(e)
