import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from mahjong_logger import encode_meld
from tenhou_to_mahjong import TenhouMeld

class TestTenhouMeldChi(unittest.TestCase):
    def test_tenhoumeld_chi(self):
        meld_dict = {
            "type": "chi",
            "m": [0, 4, 8],
            "claimed_tile": 4,
            "offset": 1
        }
        m_value = encode_meld(meld_dict)
        meld = TenhouMeld(0, m_value)
        self.assertEqual(meld.type, "chi")
        self.assertEqual(meld.from_who, 1)
        self.assertEqual(meld.base_t34, 0)
        self.assertEqual(meld.tiles_t34, [0, 1, 2])
        self.assertEqual(meld.called_index, 1)
        self.assertEqual(m_value, meld.encode())
