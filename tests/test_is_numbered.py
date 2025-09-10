import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
from mahjong_features import is_numbered

class TestIsNumbered(unittest.TestCase):
    def test_is_numbered(self):
        self.assertTrue(is_numbered(0))
        self.assertTrue(is_numbered(26))
        self.assertFalse(is_numbered(27))
        self.assertFalse(is_numbered(33))
