import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import unittest
from datetime import datetime
from gen_yama import YamaGenerator, Seed

class TestSeedToArray(unittest.TestCase):
    def test_seed_to_array(self):
        Seed._ITER = 0
        seed = Seed.generate(datetime(2020, 1, 1))
        arr = YamaGenerator(seed).seed_to_array(seed)
        self.assertEqual(len(arr), 624)
        self.assertEqual(arr[0], 1055071078)
