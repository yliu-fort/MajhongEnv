import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import unittest
from datetime import datetime
from gen_yama import Seed

class TestSeedGenerateIncrement(unittest.TestCase):
    def test_seed_generate_increment(self):
        original = Seed._ITER
        Seed._ITER = 0
        t = datetime(2020, 1, 1)
        s1 = Seed.generate(t)
        s2 = Seed.generate(t)
        self.assertNotEqual(s1, s2)
        self.assertEqual(Seed._ITER, 2)
        Seed._ITER = original
