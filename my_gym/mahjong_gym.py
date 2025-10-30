import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
import functools
import gymnasium as gym
from gymnasium.spaces import Space, Discrete, Box, MultiBinary
from pettingzoo import ParallelEnv
import numpy as np
import random
import torch
from mahjong_features import RiichiState, NUM_TILES, NUM_ACTIONS, get_action_type_from_index, NUM_FEATURES
from mahjong_features_numpy import RiichiResNetFeatures
from typing import List, Dict, Optional, Tuple, Any
from my_types import Response, PRIORITY, ActionType, Seat, ActionSketch
from mahjong_env import MahjongEnv, MahjongEnvBase
from rule_based_agent import RuleBasedAgent

AgentID = int

class MahjongEnvPettingZoo(MahjongEnv, ParallelEnv):
    """Parallel environment class.

    It steps every live agent at once. If you are unsure if you
    have implemented a ParallelEnv correctly, try running the `parallel_api_test` in
    the Developer documentation on the website.
    """

    def __init__(self, *args, **kwargs):
        ParallelEnv.__init__(self)
        
        self.metadata: dict[str, Any] = {"name": "mjai_pz_env_v0"}

        self.agents: list[AgentID] = [0, 1, 2, 3]
        self.possible_agents: list[AgentID] = [0, 1, 2, 3]
        self.observation_spaces: dict[
            AgentID, Space
        ]# = {_:Box(0, 1, (136, 34)) for _ in self.agents}  # Observation space for each agent
        self.action_spaces: dict[AgentID, Space]# = {_:Discrete(NUM_ACTIONS) for _ in self.agents}
        
        # Compatibility
        self.extractor = RiichiResNetFeatures()
        self.render_mode = 'human'
        
        MahjongEnv.__init__(self, *args, **kwargs)
    
    def reset(self, *args, **kwargs):
        self.agents = [0, 1, 2, 3]
        observations, _ = MahjongEnv.reset(self)
        for k, v in observations.items():
            v["observation"] = self.extractor(v["observation"])[0] # 136 x 34
        return observations, _

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> Space:
        """Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return gym.spaces.Dict(
            {"observation":Box(0, 1, (NUM_FEATURES, NUM_TILES)),
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
            chosen=ActionSketch(action_type=get_action_type_from_index(int(v)), payload={"action_id": int(v)})) for k, v in responses.items() if v is not None and sum(self.valid_actions[k])}
        observations, y1, y2, y3, y4 = super().step(processed_responses)
        with torch.no_grad():
            for k, v in observations.items():
                v["observation"] = self.extractor(v["observation"])[0] # 136 x 34
        if self.done == True:
            self.agents = []
        return observations, y1, y2, y3, y4

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().valid_actions


class MahjongEnvGym(MahjongEnv, gym.Env):
    """
    Gym environment class.
    """
    
    def __init__(self, imitation_reward=True, opponent_fn=None, *args, **kwargs):
        
        self.metadata: dict[str, Any] = {"name": "mjai_gym_env_v0"}

        self.observation_space: Space = gym.spaces.Dict(
            {"observation": Box(0, 1, (NUM_FEATURES, NUM_TILES)),
             "action_mask": MultiBinary(NUM_ACTIONS)}
        )
        self.action_space: Space = Discrete(NUM_ACTIONS)
        
        # Compatibility
        self.extractor = RiichiResNetFeatures()
        self.render_mode = 'human'
        
        self._focus_player = random.choice([0,1,2,3])
        self._opponent_agent = opponent_fn(self)
        self._imitation_agent = RuleBasedAgent(self)
        self._expert_instruction = None
        self._imitation_reward = imitation_reward
        self._queue = []
        self._responses = []
        self._pending_response = False
        
        MahjongEnvBase.__init__(self, *args, **kwargs)
        # 某些 gym 版本的 Env 没有 __init__；这里防御性调用
        try:
            gym.Env.__init__(self)  # 若不存在或签名不匹配，会进 except
        except Exception:
            pass
    
    def reset(self, *args, **kwargs):
        self.reward = 0.0
        MahjongEnv.reset(self)
        self._queue = []
        self._responses = []
        self._pending_response = False
        return self.step(None)[0], {}
    
    def step(self, response: Optional[int]):
        if self.done: return {}, self.reward, self.done, False, self.info

        if not self._queue:
            pending: List[Tuple[int, int, Optional[int]]] = []
            for player in range(self.num_players):
                mask = self.valid_actions[player]
                if mask is None or sum(mask[:253]) == 0:
                    self._stash_response(player, 252)
                elif sum(mask[:253]) == 1:
                    self._stash_response(player, mask.index(True))
                else:
                    pri, best_action = self._highest_priority_action(player)
                    pending.append((pri, player, best_action))
            if pending:
                pending.sort(key=lambda item: item[0], reverse=True)
                self._queue.extend(player for _, player, _ in pending)

        # 如果是focus_player则中断返回，opponents使用默认agent
        while self._queue:
            player = self._queue[0]
            mask = self.valid_actions[player]

            # 如果是
            if player == self._focus_player:
                if not self._pending_response:
                    self._pending_response = True
                    observation = {"observation": self.get_observation(player), "action_mask": mask}
                    with torch.no_grad():
                        observation["observation"] = self.extractor(observation["observation"])[0]
                        if self._imitation_reward:
                            self._expert_instruction = self._imitation_agent.predict(observation)
                    reward = self._get_and_clear_accumulated_reward(player)
                    termination = self.done
                    truncation = False
                    info = self.info
                    return observation, reward, termination, truncation, info
                else:
                    self._pending_response = False
                    
            action_idx: Optional[int] = response
            if player != self._focus_player:
                with torch.no_grad():
                    observation = {"observation": self.extractor(self.get_observation(player))[0], "action_mask": self.valid_actions[player]}
                    action_idx = self._opponent_agent.predict(observation)
            else:
                if self._imitation_reward:
                    if self._expert_instruction is not None and self._expert_instruction == action_idx:
                        self._acculmulated_rewards[player] += 0.01
                        self._expert_instruction = None

            self._stash_response(player, action_idx)
            self._queue.pop(0)
        
        assert len(self._responses) == 4, "回应不完全！"
        assert len(self._queue) == 0, "队列不为空！"

        processed_responses = {int(resp.from_seat): resp for resp in self._responses}
        self._responses = []
        self._queue = []

        super().step(processed_responses,observe=False)
        if self.done: 
            return {}, self._get_and_clear_accumulated_reward(self._focus_player), self.done, False, self.info
        
        return self.step(None)

    def action_masks(self):
        """Separate function used in order to access the action mask."""
        return self.valid_actions[self._focus_player]
    
    def _get_and_clear_accumulated_reward(self, who):
        reward = self._acculmulated_rewards[who]
        self._acculmulated_rewards[who] = 0.0
        return reward

    def _highest_priority_action(self, player: int) -> Tuple[int, Optional[int]]:
        mask = self.valid_actions[player]
        best_priority = -1
        best_action = None
        for action_idx, is_valid in enumerate(mask):
            if not is_valid:
                continue
            pri = PRIORITY.get(get_action_type_from_index(action_idx), -1)
            if pri > best_priority:
                best_priority = pri
                best_action = action_idx
        return best_priority, best_action

    def _stash_response(self, player: int, action_idx: int) -> None:
        resp = Response(
            room_id="",
            step_id=0,
            request_id="",
            from_seat=Seat(player),
            chosen=ActionSketch(
                action_type=get_action_type_from_index(action_idx),
                payload={"action_id": action_idx},
            ),
        )
        for idx, existing in enumerate(self._responses):
            if existing.from_seat == resp.from_seat:
                raise ValueError
        else:
            self._responses.append(resp)