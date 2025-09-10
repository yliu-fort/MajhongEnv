import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from mahjong_features import next_rank_wrap

class TestNextRankWrap(unittest.TestCase):
    def test_next_rank_wrap(self):
        self.assertEqual(next_rank_wrap(1), 2)
        self.assertEqual(next_rank_wrap(9), 1)
