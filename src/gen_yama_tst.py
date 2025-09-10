import hashlib
import base64
import re
import gzip
import random
from gen_yama import YamaGenerator, Seed

def _get_seed_from_plaintext_file(file):
    text = file.read()
    match_str = re.search('seed=.* ref', text).group()
    begin_pos = match_str.find(',') + 1
    end_pos = match_str.rfind('"')
    return match_str[begin_pos:end_pos]

def get_seed_from_file(filename):
    GZIP_MAGIC_NUMBER = b'\x1f\x8b'
    f = open(filename, 'rb')
    if f.read(2) == GZIP_MAGIC_NUMBER:
        f.close()
        f = gzip.open(filename, 'rt')
    else:
        f.close()
        f = open(filename, 'r')
    return _get_seed_from_plaintext_file(f)

def seed_to_array(seed_str):
    seed_bytes = base64.b64decode(seed_str)
    result = []
    for i in range(len(seed_bytes) // 4):
        result.append(int.from_bytes(seed_bytes[i*4:i*4+4], byteorder='little'))
    return result

N = 624
mt = [0] * N
mti = N + 1
def init_genrand(s):
    global mt
    global mti
    mt[0] = s & 0xffffffff
    mti = 1
    while mti < N:
        mt[mti] = (1812433253 * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti)
        mt[mti] &= 0xffffffff
        mti += 1

def init_by_array(init_key, key_length):
    global mt
    init_genrand(19650218)
    i = 1
    j = 0
    k = (N if N > key_length else key_length)
    while k != 0:
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525)) + init_key[j] + j # non linear
        mt[i] &= 0xffffffff
        i += 1
        j += 1
        if i >= N:
            mt[0] = mt[N-1]
            i = 1
        if j >= key_length:
            j = 0
        k -= 1
    k = N - 1
    while k != 0:
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941)) - i # non linear
        mt[i] &= 0xffffffff
        i += 1
        if i>=N:
            mt[0] = mt[N-1]
            i = 1
        k -= 1
    mt[0] = 0x80000000

haiDisp = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "东", "南", "西", "北", "白", "发", "中"]

def gen_yama_from_seed_tenhou(seed_str):
    seed_array = seed_to_array(seed_str)
    init_by_array(seed_array, 2496/4)
    #print(mt[623])
    mt_state = tuple(mt + [624])
    random.setstate((3, mt_state, None))
    for nKyoku in range(10):
        rnd = [0] * 144
        rnd_bytes = b''
        src = [random.getrandbits(32) for _ in range(288)]
        #print(src[0])
        for i in range(9):
            hash_source = b''
            for j in range(32):
                hash_source += src[i*32+j].to_bytes(4, byteorder='little')
            rnd_bytes += hashlib.sha512(hash_source).digest()
        for i in range(144):
            rnd[i] = int.from_bytes(rnd_bytes[i*4:i*4+4], byteorder='little')
        # till here, rnd[] has been generated
        yama = [i for i in range(136)]
        for i in range(136 - 1):
            temp = yama[i]
            yama[i] = yama[i + (rnd[i]%(136-i))]
            yama[i + (rnd[i]%(136-i))] = temp
        
        # 811注：骰子的点数对牌山无影响，只影响显示效果。也就是说，天凤永远从数组的尾部开始摸牌，
        # 数组的头部永远是王牌，并非现实中的投骰子决定哪里开始摸牌
        
        # rnd[137]～rnd[143]は未使用

        print(f'nKyoku=' + str(nKyoku) + f', yama=')
        for i in range(136):
            print(haiDisp[yama[i] // 4], end=',')
        print('')

def gen_yama_from_seed(seed_str):
    seed_array = seed_to_array(seed_str)
    init_by_array(seed_array, 2496/4)
    #print(mt[623])
    mt_state = tuple(mt + [624])
    random.setstate((3, mt_state, None))
    #for nKyoku in range(10):
    nKyoku = 0
    rnd = [0] * 144
    rnd_bytes = b''
    src = [random.getrandbits(32) for _ in range(288)]
    #print(src[0])
    for i in range(9):
        hash_source = b''
        for j in range(32):
            hash_source += src[i*32+j].to_bytes(4, byteorder='little')
        rnd_bytes += hashlib.sha512(hash_source).digest()
    for i in range(144):
        rnd[i] = int.from_bytes(rnd_bytes[i*4:i*4+4], byteorder='little')
    # till here, rnd[] has been generated
    yama = [i for i in range(136)]
    for i in range(136 - 1):
        temp = yama[i]
        yama[i] = yama[i + (rnd[i]%(136-i))]
        yama[i + (rnd[i]%(136-i))] = temp
    
    dice=(rnd[136]%6,rnd[137]%6)
    # 811注：骰子的点数对牌山无影响，只影响显示效果。也就是说，天凤永远从数组的尾部开始摸牌，
    # 数组的头部永远是王牌，并非现实中的投骰子决定哪里开始摸牌
    
    # rnd[137]～rnd[143]は未使用

    return yama

def test_mt_init():
    init_genrand(19650218)
    mt_state = tuple(mt + [624])
    random.setstate((3, mt_state, None))
    print(random.getrandbits(32))

if __name__ == "__main__":
    #test_mt_init()
    #filename = '示例饼谱.mjlog'
    #seed_str = get_seed_from_file(filename)
    seed_str="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"
    #gen_yama_from_seed(seed_str)
    assert YamaGenerator().gen_yama_from_seed(seed_str) == gen_yama_from_seed(seed_str)

    gen_yama_from_seed(Seed.generate())
    assert Seed._ITER == 1
    gen_yama_from_seed(Seed.generate())
    assert Seed._ITER == 2