import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import functools
from gymnasium.spaces import Space, Discrete
from pettingzoo import ParallelEnv

from mahjong_features import RiichiState, NUM_TILES, NUM_ACTIONS, get_action_type_from_index
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
        super(ParallelEnv, self).__init__()
        super(MahjongEnv, self).__init__(*args, **kwargs)
        
        self.metadata: dict[str, Any] = {"name": "mjai_env_v0"}

        self.agents: list[AgentID] = [0, 1, 2, 3]
        self.possible_agents: list[AgentID] = [0, 1, 2, 3]
        self.observation_spaces: dict[
            AgentID, Space
        ]  # Observation space for each agent
        self.action_spaces: dict[AgentID, Space]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> Space:
        """Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return Discrete(2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> Space:
        """Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return Discrete(NUM_ACTIONS)

    def step(self, responses: Dict[int, int]) -> Tuple[Dict[int, Dict], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict]:
        if self.agents == []: # To prevent from reset
            self.possible_agents = []
            return {}, {}, {}, {}, {}
        processed_responses = {k: Response(room_id="", step_id=0, request_id="", from_seat=Seat(k),
            chosen=ActionSketch(action_type=get_action_type_from_index(v), payload={"action_id": v})) for k, v in responses.items() if v is not None and sum(self.valid_actions[k])}
        _ = super().step(processed_responses)
        if self.done == True:
            self.agents = []
        return _