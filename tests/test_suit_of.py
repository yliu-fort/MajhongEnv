import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from mahjong_features import suit_of

class TestSuitOf(unittest.TestCase):
    def test_suit_of(self):
        self.assertEqual(suit_of(0), 0)
        self.assertEqual(suit_of(9), 1)
        self.assertEqual(suit_of(18), 2)
        self.assertIsNone(suit_of(27))
