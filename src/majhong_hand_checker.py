from collections import Counter
from mahjong.tile import TilesConverter
from mahjong.shanten import Shanten
from mahjong.agari import Agari
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.hand_calculating.hand_config import HandConfig, OptionalRules
from mahjong.meld import Meld
from mahjong.constants import EAST, SOUTH, WEST, NORTH


YAKU_JP = (
    # 1翻
    'ツモ',            # Tsumo
    'リーチ',          # Reach
    '一発',            # Ippatsu
    '槍槓',            # Chankan
    '嶺上開花',        # Rinshan-kaihou
    '海底撈月',        # Hai-tei-rao-yue
    '河底撈魚',        # Hou-tei-rao-yui
    '平和',            # Pin-fu
    '断么九',          # Tan-yao-chu
    '一盃口',          # Ii-pei-ko
    # 自風牌
    '東',              # 10:Ton
    '南',              # 11:Nan
    '西',              # 12:Xia
    '北',              # 13:Pei
    # 場風牌
    '東',              # 14:Ton
    '南',              # 15:Nan
    '西',              # 16:Xia
    '北',              # 17:Pei
    '白',              # Haku
    '發',              # Hatsu
    '中',              # Chun
    # 2翻
    'ダブルリーチ',      # Double reach
    '七対子',          # Chii-toi-tsu
    '混全帯么九',      # Chanta
    '一気通貫',        # Ikki-tsuukan
    '三色同順',        # San-shoku-dou-jun
    '三色同刻',        # San-shoku-dou-kou
    '三槓子',          # San-kan-tsu
    '対々和',          # Toi-Toi-hou
    '三暗刻',          # San-ankou
    '小三元',          # Shou-sangen
    '混老頭',          # Hon-rou-tou
    # 3翻
    '二盃口',          # Ryan-pei-kou
    '純全帯么九',      # Junchan
    '混一色',          # Hon-itsu
    # 6翻
    '清一色',          # Chin-itsu
    # 满贯
    '人和',            # Ren-hou
    # 役満
    '天和',            # Ten-hou
    '地和',            # Chi-hou
    '大三元',          # Dai-sangen
    '四暗刻',          # Suu-ankou
    '四暗刻単騎',      # Suu-ankou Tanki
    '字一色',          # Tsu-iisou
    '緑一色',          # Ryu-iisou
    '清老頭',          # Chin-routo
    '九蓮宝燈',        # Chuuren-poutou
    '純正九蓮宝燈',    # Jyunsei Chuuren-poutou 9
    '国士無双',        # Kokushi-musou
    '国士無双十三面',  # Kokushi-musou 13
    '大四喜',          # Dai-suushi
    '小四喜',          # Shou-suushi
    '四槓子',          # Su-kantsu
    # 赏牌
    'ドラ',            # Dora
    '裏ドラ',          # Ura-dora
    '赤ドラ',          # Aka-dora
)

YAKU_HAN = (
    # 1翻
    1,#'ツモ',            # Tsumo
    1,#'リーチ',          # Reach
    1,#'一発',            # Ippatsu
    1,#'槍槓',            # Chankan
    1,#'嶺上開花',        # Rinshan-kaihou
    1,#'海底撈月',        # Hai-tei-rao-yue
    1,#'河底撈魚',        # Hou-tei-rao-yui
    1,#'平和',            # Pin-fu
    1,#'断么九',          # Tan-yao-chu
    1,#'一盃口',          # Ii-pei-ko
    # 自風牌
    1,#'東',              # 10:Ton
    1,#'南',              # 11:Nan
    1,#'西',              # 12:Xia
    1,#'北',              # 13:Pei
    # 場風牌
    1,#'東',              # 14:Ton
    1,#'南',              # 15:Nan
    1,#'西',              # 16:Xia
    1,#'北',              # 17:Pei
    1,#'白',              # Haku
    1,#'發',              # Hatsu
    1,#'中',              # Chun
    # 2翻
    2,#'ダブルリーチ',      # Double reach
    2,#'七対子',          # Chii-toi-tsu
    2,#'混全帯么九',      # Chanta
    2,#'一気通貫',        # Ikki-tsuukan
    2,#'三色同順',        # San-shoku-dou-jun
    2,#'三色同刻',        # San-shoku-dou-kou
    2,#'三槓子',          # San-kan-tsu
    2,#'対々和',          # Toi-Toi-hou
    2,#'三暗刻',          # San-ankou
    2,#'小三元',          # Shou-sangen
    2,#'混老頭',          # Hon-rou-tou
    # 3翻
    3,#'二盃口',          # Ryan-pei-kou
    3,#'純全帯么九',      # Junchan
    3,#'混一色',          # Hon-itsu
    # 6翻
    6,#'清一色',          # Chin-itsu
    # 满贯
    5,#'人和',            # Ren-hou
    # 役満
    13,#'天和',            # Ten-hou
    13,#'地和',            # Chi-hou
    13,#'大三元',          # Dai-sangen
    13,#'四暗刻',          # Suu-ankou
    13,#'四暗刻単騎',      # Suu-ankou Tanki
    13,#'字一色',          # Tsu-iisou
    13,#'緑一色',          # Ryu-iisou
    13,#'清老頭',          # Chin-routo
    13,#'九蓮宝燈',        # Chuuren-poutou
    13,#'純正九蓮宝燈',    # Jyunsei Chuuren-poutou 9
    13,#'国士無双',        # Kokushi-musou
    13,#'国士無双十三面',  # Kokushi-musou 13
    13,#'大四喜',          # Dai-suushi
    13,#'小四喜',          # Shou-suushi
    13,#'四槓子',          # Su-kantsu
    # 赏牌
    0,#'ドラ',            # Dora
    0,#'裏ドラ',          # Ura-dora
    0,#'赤ドラ',          # Aka-dora
)

YAKU_HAN_FURO = (
    # 1翻
    1,#'ツモ',            # Tsumo
    1,#'リーチ',          # Reach
    1,#'一発',            # Ippatsu
    1,#'槍槓',            # Chankan
    1,#'嶺上開花',        # Rinshan-kaihou
    1,#'海底撈月',        # Hai-tei-rao-yue
    1,#'河底撈魚',        # Hou-tei-rao-yui
    1,#'平和',            # Pin-fu
    1,#'断么九',          # Tan-yao-chu
    1,#'一盃口',          # Ii-pei-ko
    # 自風牌
    1,#'東',              # 10:Ton
    1,#'南',              # 11:Nan
    1,#'西',              # 12:Xia
    1,#'北',              # 13:Pei
    # 場風牌
    1,#'東',              # 14:Ton
    1,#'南',              # 15:Nan
    1,#'西',              # 16:Xia
    1,#'北',              # 17:Pei
    1,#'白',              # Haku
    1,#'發',              # Hatsu
    1,#'中',              # Chun
    # 2翻
    2,#'ダブルリーチ',      # Double reach
    2,#'七対子',          # Chii-toi-tsu
    1,#'混全帯么九',      # Chanta
    1,#'一気通貫',        # Ikki-tsuukan
    1,#'三色同順',        # San-shoku-dou-jun
    2,#'三色同刻',        # San-shoku-dou-kou
    2,#'三槓子',          # San-kan-tsu
    2,#'対々和',          # Toi-Toi-hou
    2,#'三暗刻',          # San-ankou
    2,#'小三元',          # Shou-sangen
    2,#'混老頭',          # Hon-rou-tou
    # 3翻
    3,#'二盃口',          # Ryan-pei-kou
    2,#'純全帯么九',      # Junchan
    2,#'混一色',          # Hon-itsu
    # 6翻
    5,#'清一色',          # Chin-itsu
    # 满贯
    5,#'人和',            # Ren-hou
    # 役満
    13,#'天和',            # Ten-hou
    13,#'地和',            # Chi-hou
    13,#'大三元',          # Dai-sangen
    13,#'四暗刻',          # Suu-ankou
    13,#'四暗刻単騎',      # Suu-ankou Tanki
    13,#'字一色',          # Tsu-iisou
    13,#'緑一色',          # Ryu-iisou
    13,#'清老頭',          # Chin-routo
    13,#'九蓮宝燈',        # Chuuren-poutou
    13,#'純正九蓮宝燈',    # Jyunsei Chuuren-poutou 9
    13,#'国士無双',        # Kokushi-musou
    13,#'国士無双十三面',  # Kokushi-musou 13
    13,#'大四喜',          # Dai-suushi
    13,#'小四喜',          # Shou-suushi
    13,#'四槓子',          # Su-kantsu
    # 赏牌
    0,#'ドラ',            # Dora
    0,#'裏ドラ',          # Ura-dora
    0,#'赤ドラ',          # Aka-dora
)

YAKU_LEVEL_DICT = {
    'mangan':1,
    'haneman':2,
    'baiman':3,
    'sanbaiman':4,
    'yakuman':5
}

class MahjongHandChecker:
    MELD_TYPE_MAP = {
        "chi": Meld.CHI,
        "pon": Meld.PON,
        "kan": Meld.KAN,
        "chakan": Meld.KAN,
    }

    HAS_OPEN_TANYAO = 1 # 有食断
    HAS_AKA_DORA = 1 # 有赤宝牌

    def __init__(self):
        self.agari = Agari()
        self.shanten = Shanten()
        self.calculator = HandCalculator()

    def check_win_condition_kokushi(self, hands: list) -> bool:
        return -1 == self.shanten.calculate_shanten_for_kokushi_hand(TilesConverter.to_34_array(hands))
    
    def check_win_condition(self, hands: list, melds: list = []) -> bool:
        open_sets = [[t // 4 for t in m["m"]] for m in melds]
        for m in melds:
            for t in m["m"]:
                assert 0 <= t < 136
        return self.agari.is_agari(TilesConverter.to_34_array(hands + [t for m in melds for t in m['m']]), open_sets)

    def check_shanten(self, hands: list) -> int:
        #assert 13 <= len(tile_id) <= 14
        return self.shanten.calculate_shanten(TilesConverter.to_34_array(hands))
    
    # TODO: 检查是否有形式听牌
    def check_machii(self, hands: list, melds: list = []) -> list:
        # 返回34形式的听牌列表
        machii = []
        hands_34 = TilesConverter.to_34_array(hands + [t for m in melds for t in m['m']])
        open_sets = [[t // 4 for t in m["m"]] for m in melds]

        # 遍历所有可能的牌，计算听牌
        for tile in range(34):
            if hands_34[tile] < 4:  # 牌数量不能超过4张
                test_hand = hands_34[:]
                test_hand[tile] += 1  # 假设摸到这张牌
                if self.agari.is_agari(test_hand, open_sets):  # 检查是否能和牌
                    machii.append(tile)

        return machii

    def check_shantens(self, hands: list) -> list:
        return [self.shanten.calculate_shanten(TilesConverter.to_34_array(hands[:i]+hands[(i+1):])) for i in range(len(hands))]

    def calculate_hand_value(self, hands: list, win_tile: int, melds: list = [], config: dict = {}, raise_error=False) -> dict:
        # we had to use all 14 tiles in that array
        #tiles = TilesConverter.string_to_136_array(man='22444', pin='333567', sou='444')
        #win_tile = TilesConverter.string_to_136_array(sou='4')[0]
        #melds = [Meld(meld_type=Meld.PON, tiles=TilesConverter.string_to_136_array(man='444'))]
        tiles = hands + [t for m in melds for t in m['m']]
        melds_0 = []
        WINDS = (EAST, SOUTH, WEST, NORTH)
        
        for m in melds:
            melds_0.append(Meld(meld_type=MahjongHandChecker.MELD_TYPE_MAP[m['type']], tiles=m['m'], opened=m["opened"]))

        result = self.calculator.estimate_hand_value(tiles, \
                                                    win_tile, \
                                                    melds=melds_0, \
                                                    dora_indicators=[t for t in config.get("dora_indicators", [])], \
                                                    config=HandConfig(
                                                        is_tsumo=config.get("is_tsumo", False), \
                                                        is_riichi=config.get("is_riichi", False), \
                                                        is_ippatsu=config.get("is_ippatsu", False), \
                                                        is_rinshan=config.get("is_rinshan", False), \
                                                        is_chankan=config.get("is_chankan", False), \
                                                        is_haitei=config.get("is_haitei", False), \
                                                        is_houtei=config.get("is_houtei", False), \
                                                        is_daburu_riichi=config.get("is_daburu_riichi", False), \
                                                        is_nagashi_mangan=config.get("is_nagashi_mangan", False), \
                                                        is_tenhou=config.get("is_tenhou", False), \
                                                        is_renhou=config.get("is_renhou", False), \
                                                        is_chiihou=config.get("is_chiihou", False), \
                                                        is_open_riichi=config.get("is_open_riichi", False), \
                                                        player_wind=WINDS[config.get("player_wind", 0)], \
                                                        round_wind=WINDS[config.get("round_wind", 0)], \
                                                        kyoutaku_number=config.get("kyoutaku_number", 0), \
                                                        tsumi_number=config.get("tsumi_number", 0), \
                                                    options= \
                                                        OptionalRules( \
                                                            has_open_tanyao=MahjongHandChecker.HAS_OPEN_TANYAO, \
                                                            has_aka_dora=MahjongHandChecker.HAS_AKA_DORA, \
                                                            kiriage = False, \
                                                        )))
        if raise_error and result.error:
            raise ValueError(result.error)
        
        # Convert yaku to Japanese
        yaku = []
        yaku_tenhou = []
        yakuman_tenhou = []
        if result.yaku:
            for y in result.yaku:
                if y.tenhou_id == 52:
                    yaku.append(str(y).replace("Dora", YAKU_JP[52]))
                elif y.tenhou_id == 54:
                    yaku.append(str(y).replace("Aka Dora", YAKU_JP[54]))
                elif y.tenhou_id == 10:
                    yaku.append("自風 "+YAKU_JP[10+config.get("player_wind", 0)])
                elif y.tenhou_id == 11:
                    yaku.append("場風 "+YAKU_JP[14+config.get("round_wind", 0)])
                else:
                    yaku.append(YAKU_JP[y.tenhou_id])

                # 记录役种的天凤ID和番数（为了导出天凤格式的log）
                if YAKU_HAN[y.tenhou_id] < 13:
                    yaku_tenhou.append(y.tenhou_id)
                    yaku_tenhou.append(YAKU_HAN_FURO[y.tenhou_id] if melds else YAKU_HAN[y.tenhou_id])
                    if 52 <= y.tenhou_id <= 54:
                        yaku_tenhou[-1] = int(str(y).split(" ")[-1])
                else:
                    yakuman_tenhou.append(y.tenhou_id)

        if result.cost:
            result.cost["yaku_level"] = YAKU_LEVEL_DICT.get(result.cost["yaku_level"], 0)

        return  {   "cost":result.cost,
                    "han":result.han,
                    "fu":result.fu,
                    "fu_details":result.fu_details,
                    "yaku":yaku,
                    "yaku_tenhou":yaku_tenhou,
                    "yakuman_tenhou":yakuman_tenhou,
                    "error":result.error,
                    "is_open_hand":result.is_open_hand }


class SimpleMahjongHandChecker:
    """
    一个简单的日麻和牌判断示例类，仅用于演示基础思路。
    实际应用中还需要更多验证、役种判断等逻辑。
    """
    
    # 为了简化，这里假设牌的表示方式如下：
    # - 0~8    => 万子1~9
    # - 9~17   => 筒子1~9
    # - 18~26  => 索子1~9
    # - 27~30  => 东南西北
    # - 31~33  => 白发中
    #
    # hand 是一个长度为14的list，里面存储的是牌的编号（0~33）。
    # 在实际使用中，你可以改为更适合自己项目的表示方式，并在函数里转换即可。
    
    def check_win_condition(self, tile_id: list) -> bool:
        """
        判断一副 14 张牌是否和牌（不考虑役种、抢杠、自摸等复杂情况）。
        """
        hand = [hai//4 for hai in tile_id]

        # 先排序，方便处理
        sorted_hand = sorted(hand)
        
        # 1. 检查七对子
        if self._check_seven_pairs(sorted_hand):
            return True
        
        # 2. 检查国士无双（十三么九）
        if self._check_thirteen_orphans(sorted_hand):
            return True
        
        # 3. 检查一般型（4 面子 + 1 对子）
        if self._check_standard_pattern(sorted_hand):
            return True
        
        return False
    
    def _check_seven_pairs(self, hand: list) -> bool:
        """
        判断是否为七对子：14 张牌由 7 个对子组成。
        """
        if len(hand) != 14:
            return False
        
        # 统计每张牌出现的次数
        tile_count = Counter(hand)
        
        # 如果是七对子，则应该刚好 7 个不同的牌，而且每张牌出现 2 次
        # 在实际规则里，要额外注意“同一种牌有 4 张”这种情况（有的规则允许，有的规则不算纯正七对）
        if len(tile_count) == 7 and all(count == 2 for count in tile_count.values()):
            return True
        return False
    
    def _check_thirteen_orphans(self, hand: list) -> bool:
        """
        判断是否为国士无双（十三么九）。
        13 张么九 + 任意一张重复的么九。
        么九牌包括：万/筒/索的 1 和 9 + 东南西北中发白，共计 13 种。
        """
        if len(hand) != 14:
            return False
        
        # 么九牌的集合
        yaojiu_set = {
            0, 8,    # 万子的1、9
            9, 17,   # 筒子的1、9
            18, 26,  # 索子的1、9
            27, 28, 29, 30, 31, 32, 33  # 东南西北白发中
        }
        
        unique_tiles = set(hand)
        
        # 必须包含所有 13 种么九
        if not yaojiu_set.issubset(unique_tiles):
            return False
        
        # 再检查是不是有 14 张牌里多出了一张么九作为重复
        tile_count = Counter(hand)
        
        # 是否存在一张牌在 么九牌 集里出现 2 次
        # 且其余 12 张都是不重复的么九
        duplicated_yaojiu = any(count == 2 for tile, count in tile_count.items() if tile in yaojiu_set)
        if duplicated_yaojiu and len(unique_tiles) == 13:
            return True
        
        return False
    
    def _check_standard_pattern(self, hand: list) -> bool:
        """
        检查是否为“4 面子 + 1 对子”结构。
        实现思路：尝试从所有可能的对子中拿出做雀头，然后再判断剩余 12 张能否拆成4面子。
        """
        if len(hand) != 14:
            return False
        
        tile_count = Counter(hand)
        # 遍历所有可能的对子
        potential_pairs = [tile for tile, count in tile_count.items() if count >= 2]
        
        for pair_tile in potential_pairs:
            # 拿出这对牌，剩余的牌来判断能不能组成 4 面子
            new_hand = list(hand)
            new_hand.remove(pair_tile)
            new_hand.remove(pair_tile)
            
            if self._can_form_melds(new_hand):
                return True
        return False

    def _can_form_melds(self, tiles: list) -> bool:
        if not tiles:
            return True

        tiles.sort()
        first_tile = tiles[0]

        from collections import Counter
        counter = Counter(tiles)

        # 1. 尝试刻子（3 张相同的牌）
        if counter[first_tile] >= 3:
            new_tiles = list(tiles)
            new_tiles.remove(first_tile)
            new_tiles.remove(first_tile)
            new_tiles.remove(first_tile)
            if self._can_form_melds(new_tiles):
                return True

        # 2. 尝试顺子（仅数牌且三张牌必须同花色）
        #    先排除字牌 >= 27
        if first_tile < 27:
            # 检查花色：如果 first_tile // 9 == second_tile // 9 == third_tile // 9，才算同花色
            second_tile = first_tile + 1
            third_tile = first_tile + 2

            # 同花色检查
            if (second_tile < 27 and third_tile < 27 and
                first_tile // 9 == second_tile // 9 == third_tile // 9):
                
                if counter[second_tile] > 0 and counter[third_tile] > 0:
                    new_tiles = list(tiles)
                    new_tiles.remove(first_tile)
                    new_tiles.remove(second_tile)
                    new_tiles.remove(third_tile)
                    if self._can_form_melds(new_tiles):
                        return True

        return False


# ------------------ 测试示例 ------------------ #
if __name__ == "__main__":
    from mahjong_tiles_print_style import tiles_printout
    checker = MahjongHandChecker()
    simple_checker = SimpleMahjongHandChecker()
    agari = Agari()
    
    # 一个简单示例：假设这是一个标准的 4 面子 + 1 对子手牌
    # 例如：111 万, 234 索, 789 筒, 东东东（刻子）+ 对子白白
    # 下面举例手牌：111万 (0,0,0), 234索(19,20,21), 789筒(15,16,17)，东东东(27,27,27)，白白(31,31)
    # 具体数字仅示例，不一定完全对应真实牌型。
    
    sample_hand = [0, 0, 0, 18, 19, 20, 15, 16, 17, 27, 27, 27, 31, 31]
    result = checker.check_win_condition(sample_hand)
    print(tiles_printout([hai*4 for hai in sample_hand]), "和牌判断结果:", result)

    sample_hand = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8]
    sample_hand = [x*4 for x in sample_hand]
    result = checker.check_win_condition(sample_hand)
    print(tiles_printout(sample_hand), "和牌判断结果:", result)

    sample_hand = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 9]
    sample_hand = [x*4 for x in sample_hand]
    result = checker.check_shanten(sample_hand)
    print("手牌", tiles_printout(sample_hand), "上听数:", result)

    def _string_to_open_34_set(sou="", pin="", man="", honors=""):
        open_set = TilesConverter.string_to_136_array(sou=sou, pin=pin, man=man, honors=honors)
        open_set[0] //= 4
        open_set[1] //= 4
        open_set[2] //= 4
        return open_set

    tiles = TilesConverter.string_to_34_array(sou="23455567", pin="222", man="345")
    melds = [
        _string_to_open_34_set(man="345"),
        _string_to_open_34_set(sou="555"),
    ]
    result = agari.is_agari(tiles, melds)
    print(result, tiles, melds)

    sample_hand = TilesConverter.string_to_136_array(sou="23467", pin="222", man="345")
    open_set = [{"m": [89, 90, 88]}]
    #open_set = [{"m": [2*4, 3*4, 4*4]}, {"m": [22*4, 22*4+1, 22*4+2]}]
    print("手牌", tiles_printout(sample_hand), "副露:", *[tiles_printout(o["m"]) for o in open_set])
    result = checker.check_win_condition(sample_hand, open_set)
    print(tiles_printout(sample_hand), *[tiles_printout(o["m"]) for o in open_set], "和牌判断结果:", result)

    sample_hand = [62, 8, 25, 70, 60, 78, 1, 76, 56, 22, 115, 122, 123, 120]
    sample_hand = TilesConverter.to_34_array(sample_hand)
    open_set = [[122//4, 123//4, 120//4]]
    result = agari.is_agari(sample_hand, open_set)
    print("副露测试例1", result, tiles, melds)

    sample_hand = [62, 8, 25, 70, 60, 78, 1, 76, 56, 22, 115]
    open_set = [{"m": [122, 123, 120]}]
    print("手牌", tiles_printout(sample_hand), "副露:", tiles_printout(open_set[0]["m"]))
    result = checker.check_win_condition(sample_hand, open_set)
    print(tiles_printout(sample_hand), tiles_printout(open_set[0]["m"]), "和牌判断结果:", result)

    sample_hand = [34, 20, 76, 4, 54, 71, 95, 116, 1, 100, 48]
    open_set = [{"m": [127, 125, 126]}]
    result = checker.check_win_condition(sample_hand, open_set)
    print(tiles_printout(sample_hand), tiles_printout(open_set[0]["m"]), "和牌判断结果:", result)

    sample_hand = TilesConverter.string_to_136_array(sou="33", pin="145", man="", honors="1234")
    result = checker.check_win_condition_kokushi(sample_hand)
    print("国士未和了手牌", tiles_printout(sample_hand), "和了:", result)

    sample_hand = TilesConverter.string_to_136_array(sou="19", pin="19", man="199", honors="1234567")
    result = checker.check_win_condition_kokushi(sample_hand)
    print("国士和了手牌", tiles_printout(sample_hand), "和了:", result)

    # 和牌点数计算
    tiles = TilesConverter.string_to_136_array(man='22444', pin='333567', sou='555')
    win_tile = TilesConverter.string_to_136_array(sou='5')[0]
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": False, "player_wind": 0, "round_wind": 0,})
    print("和牌点数役种计算", result)

    # 和牌点数计算
    tiles = TilesConverter.string_to_136_array(man='22444', pin='333567', sou='555')
    win_tile = TilesConverter.string_to_136_array(sou='5')[0]
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": True, "player_wind": 0, "round_wind": 0,})
    print("和牌点数役种计算", result)

    # 和牌点数计算
    tiles = TilesConverter.string_to_136_array(man='22444', pin='333567', sou='555')
    win_tile = TilesConverter.string_to_136_array(sou='5')[0]
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": False, "player_wind": 0, "round_wind": 0, "kyoutaku_number": 3, "tsumi_number": 2})
    print("和牌点数役种计算+供托", result)

    # 和牌点数计算
    tiles = TilesConverter.string_to_136_array(man='22444', pin='333567', sou='555')
    win_tile = TilesConverter.string_to_136_array(sou='5')[0]
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": True, "player_wind": 0, "round_wind": 0, "kyoutaku_number": 3, "tsumi_number": 2})
    print("和牌点数役种计算+供托", result)

    # 和牌点数计算
    tiles = TilesConverter.string_to_136_array(man='22444', pin='333567', sou='555')
    win_tile = TilesConverter.string_to_136_array(sou='5')[0]
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": False, "player_wind": 0, "round_wind": 0, "dora_indicators": TilesConverter.string_to_136_array(man='3', pin='2', sou='4')})
    print("和牌点数役种计算+宝牌", result)

    # 和牌点数计算
    tiles = TilesConverter.string_to_136_array(man='22444', pin='333567', sou='555')
    win_tile = TilesConverter.string_to_136_array(sou='5')[0]
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": True, "player_wind": 0, "round_wind": 0, "dora_indicators": TilesConverter.string_to_136_array(man='3', pin='2', sou='4')})
    print("和牌点数役种计算+宝牌", result)

    # 和牌点数计算: 自风与场风
    tiles = TilesConverter.string_to_136_array(man='22444', pin='567', honors='333444')
    win_tile = TilesConverter.string_to_136_array(man='2')[0]
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": True, "player_wind": 2, "round_wind": 3})
    print("和牌点数役种计算: 自风与场风 ", result)

    # 无役情况
    tiles = TilesConverter.string_to_136_array(man='11444', pin='333567', sou='567')
    win_tile = TilesConverter.string_to_136_array(sou='5')[0]
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": False, "player_wind": 0, "round_wind": 0})
    print("无役 ", result)

    # 错牌情况
    tiles = [0, 0, 12, 13, 14, 44, 45, 46, 52, 56, 60, 88, 92, 96]
    win_tile = 88
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": False, "player_wind": 0, "round_wind": 0})
    print("错牌 ", result)

    tiles = [0, 0, 12, 13, 14, 44, 45, 46, 52, 56, 60, 88, 92, 96]
    win_tile = 0
    result = checker.calculate_hand_value(tiles, win_tile, [], {"is_tsumo": False, "player_wind": 0, "round_wind": 0})
    print("错牌 ", result)

    # 和牌点数计算 庄家立直自摸
    tiles = [87,40,57,60,51,66,84,42,10,43,86,15,18,44]
    win_tile = 44
    result = checker.calculate_hand_value(tiles, win_tile, [], {"dora_indicators": [38, 97], "is_riichi": True, "is_tsumo": True, "player_wind": 0, "round_wind": 0, "kyoutaku_number": 1, "tsumi_number": 0})
    print("和牌点数役种计算 庄家立直自摸", result)