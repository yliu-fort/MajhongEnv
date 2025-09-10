import os, sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from mahjong_tiles_print_style import MahjongTileStyle


class TestMahjongTileStyle(unittest.TestCase):
    def test_get_tile_printout(self):
        self.assertEqual(MahjongTileStyle.get_tile_printout(0), "🀇 ")

    def test_get_tiles_printout(self):
        self.assertEqual(MahjongTileStyle.get_tiles_printout([0, 4, 8]), "🀇 🀈 🀉 ")


if __name__ == "__main__":
    unittest.main()

