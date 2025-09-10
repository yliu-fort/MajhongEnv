import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import unittest
from mahjong_tiles_print_style import tile_printout

class TestTilePrintout(unittest.TestCase):
    def test_tile_printout(self):
        self.assertEqual(tile_printout(-1), "🀫 ")
