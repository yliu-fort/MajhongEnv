import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from tenhou_to_mahjong import tid136_to_t34

class TestTid136ToT34(unittest.TestCase):
    def test_tid136_to_t34(self):
        self.assertEqual(tid136_to_t34(0), 0)
        self.assertEqual(tid136_to_t34(4), 1)
        self.assertEqual(tid136_to_t34(135), 33)
