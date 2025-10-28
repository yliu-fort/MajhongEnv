import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import functools
import gymnasium as gym
from gymnasium.spaces import Space, Discrete, Box, MultiBinary
from pettingzoo import ParallelEnv
import numpy as np
from mahjong_features import RiichiState, NUM_TILES, NUM_ACTIONS, get_action_type_from_index
from mahjong_features_numpy import RiichiResNetFeatures
from typing import List, Dict, Optional, Tuple, Any
from my_types import Response, PRIORITY, ActionType, Seat, ActionSketch
from mahjong_env import MahjongEnv


AgentID = int

class MahjongEnvPettingZoo(MahjongEnv, ParallelEnv):
    """Parallel environment class.

    It steps every live agent at once. If you are unsure if you
    have implemented a ParallelEnv correctly, try running the `parallel_api_test` in
    the Developer documentation on the website.
    """

    def __init__(self, *args, **kwargs):
        ParallelEnv.__init__(self)
        
        self.metadata: dict[str, Any] = {"name": "mjai_env_v0"}

        self.agents: list[AgentID] = [0, 1, 2, 3]
        self.possible_agents: list[AgentID] = [0, 1, 2, 3]
        self.observation_spaces: dict[
            AgentID, Space
        ]# = {_:Box(0, 1, (136, 34)) for _ in self.agents}  # Observation space for each agent
        self.action_spaces: dict[AgentID, Space]# = {_:Discrete(NUM_ACTIONS) for _ in self.agents}
        
        # Compatibility
        #self.observation_space = lambda _: self.observation_spaces[_]
        #self.action_space = lambda _: self.action_spaces[_]
        
        self.extractor = RiichiResNetFeatures()
        self.render_mode = 'human'
        
        MahjongEnv.__init__(self, *args, **kwargs)
    
    def reset(self, *args, **kwargs):
        self.agents = [0, 1, 2, 3]
        observations, _ = MahjongEnv.reset(self)
        for k, v in observations.items():
            v["observation"] = self.extractor(v["observation"])[0] # 136 x 34
            #v["action_mask"] = np.asarray(v["action_mask"],dtype=np.int8)[None,:]
        return observations, _

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> Space:
        """Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return gym.spaces.Dict(
            {"observation":Box(0, 1, (136, 34)),
             "action_mask": MultiBinary(NUM_ACTIONS)}
        )
    
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> Space:
        """Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return Discrete(NUM_ACTIONS)

    def step(self, responses: Dict[int, int]) -> Tuple[Dict[int, Dict], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict]:
        if self.agents == []: # To prevent from reset
            return {}, {}, {}, {}, {}
        processed_responses = {k: Response(room_id="", step_id=0, request_id="", from_seat=Seat(k),
            chosen=ActionSketch(action_type=get_action_type_from_index(v), payload={"action_id": v})) for k, v in responses.items() if v is not None and sum(self.valid_actions[k])}
        observations, y1, y2, y3, y4 = super().step(processed_responses)
        for k, v in observations.items():
            v["observation"] = self.extractor(v["observation"])[0] # 136 x 34
            #v["action_mask"] = np.asarray(v["action_mask"],dtype=np.int8)[None,:]
        if self.done == True:
            self.agents = []
        return observations, y1, y2, y3, y4

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().valid_actions