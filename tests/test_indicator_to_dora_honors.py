import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from mahjong_features import indicator_to_dora

class TestIndicatorToDoraHonors(unittest.TestCase):
    def test_indicator_to_dora_honors(self):
        self.assertEqual(indicator_to_dora(27), 28)
        self.assertEqual(indicator_to_dora(30), 27)
        self.assertEqual(indicator_to_dora(31), 32)
        self.assertEqual(indicator_to_dora(33), 31)
