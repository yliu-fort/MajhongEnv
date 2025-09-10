import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from mahjong_logger import encode_meld
from tenhou_to_mahjong import TenhouMeld

class TestTenhouMeldChakan(unittest.TestCase):
    def test_tenhoumeld_chakan(self):
        meld_dict = {
            "type": "chakan",
            "m": [0, 1, 2, 3],
            "claimed_tile": 0,
            "offset": 2
        }
        m_value = encode_meld(meld_dict)
        meld = TenhouMeld(0, m_value)
        self.assertEqual(meld.type, "chakan")
        self.assertEqual(meld.from_who, 2)
        self.assertEqual(meld.base_t34, 0)
        self.assertEqual(meld.tiles_t34, [0, 0, 0, 0])
        self.assertEqual(meld.called_index, 0)
