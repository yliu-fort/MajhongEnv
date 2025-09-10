import hashlib
import base64
import re
import gzip
import random
from datetime import datetime


# Generate random seed for yama generator
class Seed:
    _ITER = 0

    @staticmethod
    def generate(current_time = datetime.now()):
        Seed._ITER += 1
        SEED_DICT = (
        "立直", "門前清自摸和", "断幺九", "一盃口", "平和", "自摸", "役牌", "槓", "暗槓", "明槓", 
        "嶺上開花", "海底撈月", "河底撈魚", "七対子", "混全帯幺九", "純全帯幺九", "混一色", "清一色", 
        "三色同順", "一気通貫", "対々和", "三暗刻", "三槓子", "混老頭", "小三元", "大三元", "小四喜", 
        "大四喜", "字一色", "緑一色", "清老頭", "四暗刻", "国士無双", "国士無双十三面", "九蓮宝燈", "四槓子", "天和", "地和", "人和"
        )
        seeds = []
        for s in SEED_DICT:
            seed = s + current_time.strftime('%Y%m%d%H%M%S%f') + str(62208 + Seed._ITER) # Use timestamp with microsecond precision
            seed_hash = hashlib.sha512(seed.encode('utf-8')).digest()
            seeds.append(seed_hash)
        seed_str = base64.b64encode(b''.join(seeds)).decode('utf-8')
        return seed_str
    

# Generate yama from seed
class YamaGenerator:
    hai_disp = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", 
                    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", 
                    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", 
                    "东", "南", "西", "北", "白", "发", "中")
    
    def __init__(self, seed = None):
        self.seed_str = seed or Seed.generate()

        self.N = 624
        self.mt = [0] * self.N
        self.mti = self.N + 1

        self.reset_random_state()

    def _get_seed_from_plaintext_file(self, file):
        text = file.read()
        match_str = re.search(r'seed=.* ref', text).group()
        begin_pos = match_str.find(',') + 1
        end_pos = match_str.rfind('"')
        return match_str[begin_pos:end_pos]

    def get_seed_from_file(self, filename):
        GZIP_MAGIC_NUMBER = b'\x1f\x8b'
        with open(filename, 'rb') as f:
            if f.read(2) == GZIP_MAGIC_NUMBER:
                f = gzip.open(filename, 'rt')
            else:
                f = open(filename, 'r')
            return self._get_seed_from_plaintext_file(f)

    def seed_to_array(self, seed_str):
        seed_bytes = base64.b64decode(seed_str)
        result = []
        for i in range(len(seed_bytes) // 4):
            result.append(int.from_bytes(seed_bytes[i*4:i*4+4], byteorder='little'))
        return result

    def init_genrand(self, s):
        self.mt[0] = s & 0xffffffff
        self.mti = 1
        while self.mti < self.N:
            self.mt[self.mti] = (1812433253 * (self.mt[self.mti - 1] ^ (self.mt[self.mti - 1] >> 30)) + self.mti)
            self.mt[self.mti] &= 0xffffffff
            self.mti += 1

    def init_by_array(self, init_key, key_length):
        self.init_genrand(19650218)
        i, j, k = 1, 0, max(self.N, key_length)

        while k != 0:
            self.mt[i] = (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1664525)) + init_key[j] + j
            self.mt[i] &= 0xffffffff
            i += 1
            j += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
            if j >= key_length:
                j = 0
            k -= 1

        k = self.N - 1
        while k != 0:
            self.mt[i] = (self.mt[i] ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) * 1566083941)) - i
            self.mt[i] &= 0xffffffff
            i += 1
            if i >= self.N:
                self.mt[0] = self.mt[self.N - 1]
                i = 1
            k -= 1

        self.mt[0] = 0x80000000

    def reset_random_state(self):
        seed_array = self.seed_to_array(self.seed_str)
        self.init_by_array(seed_array, len(seed_array))
        mt_state = tuple(self.mt + [624])
        random.setstate((3, mt_state, None))
    
    def generate(self):
        rnd = [0] * 144
        rnd_bytes = b''
        src = [random.getrandbits(32) for _ in range(288)]

        for i in range(9):
            hash_source = b''.join(src[i * 32 + j].to_bytes(4, byteorder='little') for j in range(32))
            rnd_bytes += hashlib.sha512(hash_source).digest()

        rnd = [int.from_bytes(rnd_bytes[i * 4:i * 4 + 4], byteorder='little') for i in range(144)]

        yama = [i for i in range(136)]
        for i in range(135):
            swap_idx = i + (rnd[i] % (136 - i))
            yama[i], yama[swap_idx] = yama[swap_idx], yama[i]
        
        dice = (rnd[135] % 6, rnd[136] % 6)

        return yama, dice


if __name__ == "__main__":
    example_seed = "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"
    # kyoku 1 oya="0" 
    # hai0="116,86,99,6,36,40,97,83,77,55,57,24,63" 
    # hai1="106,43,20,58,45,127,134,61,112,70,23,42,35" 
    # hai2="118,18,14,32,22,120,96,2,130,27,64,109,0" 
    # hai3="126,82,123,31,1,94,48,26,52,95,78,119,16"

    # kyoku 4 oya="1" 
    # hai0="45,1,73,24,110,74,0,89,104,58,51,69,6" 
    # hai1="62,108,68,122,7,91,19,49,50,22,134,40,84" 
    # hai2="132,54,17,41,10,119,111,56,20,23,33,114,32" 
    # hai3="127,115,72,35,67,57,107,48,102,59,117,121,101"
    gen = YamaGenerator(seed=example_seed)
    yama1, dice1 = gen.generate()
    yama2, dice2 = gen.generate()
    yama3, dice3 = gen.generate()
    yama4, dice4 = gen.generate()

    assert yama1[5] == 51
    assert dice1 == (2, 0)
    assert yama2[5] == 62
    assert dice2 == (2, 2)

    #print(yama1)
    #print(yama4)