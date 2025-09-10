import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from mahjong_features import indicator_to_dora

class TestIndicatorToDoraNumbered(unittest.TestCase):
    def test_indicator_to_dora_numbered(self):
        self.assertEqual(indicator_to_dora(0), 1)
        self.assertEqual(indicator_to_dora(8), 0)
        self.assertEqual(indicator_to_dora(9), 10)
        self.assertEqual(indicator_to_dora(26), 18)
