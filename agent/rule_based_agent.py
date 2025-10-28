from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from shanten_dp import compute_ukeire_advanced
from random_discard_agent import RandomDiscardAgent
from mahjong_features import RiichiResNetFeatures, NUM_TILES, get_action_type_from_index, NUM_ACTIONS
from mahjong.tile import TilesConverter
from mahjong_tiles_print_style import tile_printout
from my_types import Response, ActionSketch, Seat, ActionType
from typing import Any

def tid136_to_t34(tid: int) -> int:
    return tid // 4

def good_moves(hand_34, remaining):
    # Return list of discards that drop shanten and maximize ukeire.
    # If none can lower shanten further than others, return those with max ukeire.
    best_discards = []
    best_shanten = 1_000_000
    best_ukeire = -1

    for i, cnt in enumerate(hand_34):
        if cnt <= 0:
            continue
        out = compute_ukeire_advanced(hand_34, i, remaining)
        s = out.get("shanten", 1_000_000)

        if s < best_shanten:
            best_shanten = s
            best_discards = [(i,out)]
        elif s == best_shanten:
            best_discards.append((i,out))
    
    # sort best_discards by ukeire (descending)
    if best_discards:
        best_discards.sort(key=lambda x: x[1].get("ukeire", -1), reverse=True)

    return best_discards


class RuleBasedAgent:
    def __init__(self, env: Any, backbone: str = ""):
        self.env = env
        self._alt_model = RandomDiscardAgent(env)
        self.extractor = RiichiResNetFeatures()
 
    def train(self, total_timesteps=100000):
        pass
 
    def save_model(self, path=""):
        pass
 
    def load_model(self, path=""):
        pass

    
    def predict(self, observation) -> ActionSketch:
        #who = observation[1]["who"]
        action_masks = observation.legal_actions_mask
        valid_action_list = [i for i in list(range(NUM_ACTIONS)) if action_masks[i]]
        allowed_action_type = set([get_action_type_from_index(i) for i, a in enumerate(action_masks) if a])
        #print(allowed_action_type)
        if sum(action_masks) == 0:
            return ActionSketch(action_type=ActionType.UNKNOWN, payload={"action_id": 0})
        elif sum(action_masks) == 1:
            action_id = valid_action_list[0]
            return ActionSketch(action_type=get_action_type_from_index(action_id), payload={"action_id": action_id})

        if self.env and (any([_ in allowed_action_type for _ in [ActionType.TSUMO,]])):
            return ActionSketch(action_type=ActionType.TSUMO, payload={"action_id": 251})

        if self.env and (any([_ in allowed_action_type for _ in [ActionType.RON,]])):
            return ActionSketch(action_type=ActionType.RON, payload={"action_id": 250})

        if self.env and (any([_ in allowed_action_type for _ in [ActionType.RYUUKYOKU,]])):
            return ActionSketch(action_type=ActionType.RYUUKYOKU, payload={"action_id": 249})
               
        # 如果当前状态是和牌状态，直接返回和牌动作        
        if self.env and (ActionType.DISCARD in allowed_action_type):
            # 推理时获取动作
            state = observation
            shantens = state.shantens
            ukeires = state.ukeires
            out = self.extractor(state)
            x = out["x"][:,:,0].numpy()
            m = out["legal_mask"].numpy()
            #hands_34 = (np.array(self.env.hands[who]) // 4).tolist()

            discard_priority_attack = sorted(list(range(NUM_TILES)), key=lambda i: (-int(m[i]), shantens[i], -ukeires[i], i))
            if m[discard_priority_attack[0]] == 1:
                action_id = discard_priority_attack[0]
                action_id = action_id+34 if action_masks[action_id+34] else action_id
                return ActionSketch(action_type=get_action_type_from_index(action_id), payload={"action_id": action_id})

        if self.env and (any([_ in allowed_action_type for _ in [ActionType.CHI,]])):
            return ActionSketch(action_type=ActionType.PASS, payload={"action_id": 252})
        
        # if preds not in action_masks, return a random choice from action_masks.
        if self.env and (any([_ in allowed_action_type for _ in [ActionType.PON, ActionType.KAN]])):
            state = observation
            
            remaining = RiichiResNetFeatures._default_visible_counts(state)
            claim = self.env.claims[0] # e.g., {"type": "pon", "fromwho": player, "who": other_player, "tile": tile})
            hand_counts = state.hand_counts[:] # should be 4, 7, 10, 13
            base_sh = good_moves(hand_counts, remaining)[0][1]['shanten']

            # get valid moves
            if ActionType.KAN in allowed_action_type:
                hand_counts[claim["tile"]//4]-=3
            elif ActionType.PON in allowed_action_type:
                hand_counts[claim["tile"]//4]-=2

            new_sh = good_moves(hand_counts, remaining)[0][1]['shanten']
            turn_number = state.turn_number
            should_call = ( new_sh < base_sh ) & (turn_number >= 6) & (base_sh > 2)

            return self._alt_model.predict(observation) if should_call \
                else ActionSketch(action_type=ActionType.PASS, payload={"action_id": 252})
        
        return self._alt_model.predict(observation)


if __name__ == "__main__":
    def display_good_moves(tile):
        hand = tile[:-2]
        draw = tile[-2:]
        hand_34 = TilesConverter.one_line_string_to_34_array(hand)
        draw_34 = TilesConverter.one_line_string_to_34_array(draw)
        all_34 = (np.array(hand_34)+np.array(draw_34)).tolist()
        [print(tile_printout(m[0]*4+1), m[1]) for m in good_moves(all_34, remaining = [4-x for x in all_34])]
        print("***")

    def display_good_calls(hand):
        hand_34 = TilesConverter.one_line_string_to_34_array(hand)
        all_34 = (np.array(hand_34)).tolist()
        [print(tile_printout(m[0]*4+1), m[1]) for m in good_moves(all_34, remaining = [4-x for x in all_34])]
        print("***")

    display_good_moves("3679m1779p156s57z9p")
    display_good_moves("10m2299p3467s555z1s")
    display_good_moves("1223556778899s4s")
    display_good_moves("13457m123p3489s1z7m")
    display_good_moves("34577m123p3489s1z7s")
    display_good_calls("34577m123p3489s1z")
    display_good_calls("34577m123p34s1z")
