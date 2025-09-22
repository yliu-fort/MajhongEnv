from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import gymnasium as gym
import numpy as np

from shanten_dp import compute_ukeire_advanced
from random_discard_agent import RandomDiscardAgent
from mahjong_features import RiichiResNetFeatures
from mahjong.tile import TilesConverter
from mahjong_tiles_print_style import tile_printout

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


class HaiEfficiencyAgent:
    def __init__(self, env: gym.Env, backbone: str = ""):
        self.env = env
        self._alt_model = RandomDiscardAgent(env)
 
    def train(self, total_timesteps=100000):
        pass
 
    def save_model(self, path=""):
        pass
 
    def load_model(self, path=""):
        pass

    
    def predict(self, observation):
        # 如果当前状态是和牌状态，直接返回和牌动作
        if self.env and (self.env.phase == "tsumo" or self.env.phase == "ron"):
            return (0, True)
        
        # No options yet
        if self.env and (self.env.phase == "riichi"):
            return self._alt_model.predict(observation)[0], True if self.env.num_riichi < 3 else False

        
        if self.env and (self.env.phase == "discard"):
            # 推理时获取动作
            state = observation[0]
            shantens = state.shantens
            ukeires = state.ukeires
            #print(state.shantens)

            best_shanten = 255
            best_ukeire = -1

            # Check if pred falls in valid action_masks
            action_masks = self.env.action_masks() # 0 - 13 position in hand
            for i, x in enumerate(self.env.hands[observation[1]['who']]):
                if action_masks[i] == True:
                    t_34 = tid136_to_t34(x)
                    sh = shantens[t_34]
                    uke = ukeires[t_34]
                    if sh < best_shanten or (sh == best_shanten and uke >= best_ukeire):
                        best_shanten = sh
                        best_ukeire = uke
                        pred = t_34

            for i, x in enumerate(self.env.hands[observation[1]['who']]):
                if tid136_to_t34(x) == pred and action_masks[i] == True:
                    return (i, False)

        # if preds not in action_masks, return a random choice from action_masks.
        if self.env and (self.env.phase in ["pon", "kan"]):
            state = observation[0]
            remaining = RiichiResNetFeatures._default_visible_counts(state)
            claim = self.env.claims[0] # e.g., {"type": "pon", "fromwho": player, "who": other_player, "tile": tile})
            hand_counts = state.hand_counts[:] # should be 4, 7, 10, 13
            
            base_sh = good_moves(hand_counts, remaining)[0][1]['shanten']
            # get valid moves
            if self.env.phase == "pon":
                hand_counts[claim["tile"]//4]-=2
            elif self.env.phase == "kan":
                hand_counts[claim["tile"]//4]-=3

            new_sh = good_moves(hand_counts, remaining)[0][1]['shanten']
            turn_number = state.turn_number
            should_call = ( new_sh < base_sh ) & (turn_number >= 6) & (base_sh > 2)

            return self._alt_model.predict(observation)[0], should_call

        return self._alt_model.predict(observation)[0], False


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
