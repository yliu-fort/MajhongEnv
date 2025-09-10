import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from mahjong_features import rank_of

class TestRankOf(unittest.TestCase):
    def test_rank_of(self):
        self.assertEqual(rank_of(0), 1)
        self.assertEqual(rank_of(8), 9)
        self.assertEqual(rank_of(9), 1)
        self.assertEqual(rank_of(17), 9)
        self.assertEqual(rank_of(18), 1)
        self.assertEqual(rank_of(26), 9)
        self.assertIsNone(rank_of(27))
