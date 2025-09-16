import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from mahjong.tile import TilesConverter
from shanten_dp import compute_ukeire_advanced
from mahjong.shanten import Shanten

class TestShantenDP(unittest.TestCase):
    sh = Shanten()

    def _test_shanten_fn(self, tile: str, shanten: int):
        hand = tile[:-2]
        draw = tile[-2:]
        hand_34 = TilesConverter.one_line_string_to_34_array(hand)
        draw_34 = TilesConverter.one_line_string_to_34_array(draw)
        all_34 = [x + y for x, y in zip(hand_34, draw_34)]
        last_draw = next(i for i, v in enumerate(draw_34) if v)
        out = compute_ukeire_advanced(all_34, last_draw, remaining = [4-x for x in all_34])
        self.assertEqual(out['shanten'], shanten)

    def _test_ukeire_fn(self, tile: str, ukeire: int):
        hand = tile[:-2]
        draw = tile[-2:]
        hand_34 = TilesConverter.one_line_string_to_34_array(hand)
        draw_34 = TilesConverter.one_line_string_to_34_array(draw)
        all_34 = [x + y for x, y in zip(hand_34, draw_34)]
        last_draw = next(i for i, v in enumerate(draw_34) if v)
        out = compute_ukeire_advanced(all_34, last_draw, remaining = [4-x for x in all_34])
        self.assertEqual(out['ukeire'], ukeire)

    def _test_shanten_fn2(self, tile: str, shanten: int):
        hand = tile[:-2]
        draw = tile[-2:]
        hand_34 = TilesConverter.one_line_string_to_34_array(hand)
        draw_34 = TilesConverter.one_line_string_to_34_array(draw)
        out = self.sh.calculate_shanten(hand_34 + draw_34)
        self.assertEqual(out, shanten)

    def test_1(self):
       #self._test_shanten_fn(tile="67m140999p37s246z7p", shanten=4)
       self._test_shanten_fn(tile="89m489p589s14466z3m", shanten=3)
       #self._test_shanten_fn(tile="68m79p110679s366z4p", shanten=2)
       self._test_shanten_fn(tile="23p48s7p", shanten=1) #一向听
       self._test_shanten_fn(tile="7p4478s36z4m", shanten=2) #二向听
       self._test_shanten_fn(tile="9m2377p12s566z9s", shanten=2) #二向听

    def test_2(self):
       #self._test_ukeire_fn(tile="67m140999p37s246z7p", ukeire=66)
       self._test_ukeire_fn(tile="89m489p589s14466z3m", ukeire=16)
       #self._test_ukeire_fn(tile="8m479p110679s366z6m", ukeire=12)

    def test_3(self):
       #self. _test_shanten_fn2(tile="67m140999p37s246z7p", shanten=4)
       self. _test_shanten_fn2(tile="89m489p589s14466z", shanten=3)
       self. _test_shanten_fn2(tile="68m79p110679s366z", shanten=2)
       self. _test_shanten_fn2(tile="23p48s7p", shanten=1)
