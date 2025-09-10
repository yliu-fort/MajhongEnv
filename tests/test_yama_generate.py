import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import unittest
from datetime import datetime
from gen_yama import YamaGenerator, Seed

class TestYamaGenerate(unittest.TestCase):
    def test_yama_generate(self):
        Seed._ITER = 0
        seed = Seed.generate(datetime(2020, 1, 1))
        gen = YamaGenerator(seed=seed)
        yama, dice = gen.generate()
        self.assertEqual(len(yama), 136)
        self.assertEqual(dice, (2, 3))
        self.assertEqual(yama[5], 89)
