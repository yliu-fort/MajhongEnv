import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import Counter
from random import randint
from gen_yama import YamaGenerator
from mahjong_tiles_print_style import tile_printout, tiles_printout
from mahjong_hand_checker import MahjongHandChecker
from mahjong_logger import MahjongLogger
from agent.random_discard_agent import RandomDiscardAgent


class MahjongEnvBase(gym.Env):
    """
    一个简化的麻将环境示例。
    """
    # 用于将动作映射到具体的操作
    PHASE_MAP = {
        "draw": 0,
        "kan_draw": 0,
        "discard": 1,
        "chi": 2,
        "pon": 3,
        "kan": 4,
        "chakan": 5,
        "ankan": 6,
        "riichi": 7,
        "ryuukyoku": 8,
        "tsumo": 9,
        "ron": 10,
        "score": 11,
        "game_over": 11,
    }

    HAS_KUIKAE = 0 # 食替 (TODO: 无实现)
    HAS_NAKU = 1 # 鸣牌
    HAS_FURITEN = 1 # 振听
    KYOUTAKU = 10 # 供托
    TSUMI = 1 # 积
    RYUUKYOKU_PENALTY = 30 # 流局罚符，标准为3000点
    MAX_ROUND = 16 # 最大局数
    MAX_ROUND_EXTRA = 4 # 最大延长战局数 (TODO: 无实现)
    MAX_HONBA = 8 # 最大本场数

    def __init__(self, num_players=4, num_rounds=4):
        super(MahjongEnvBase, self).__init__()
        
        # 设置玩家数量
        self.num_players = num_players

        # 设置局数, 4: 东风场, 8: 半庄， 16: 全庄
        self.num_rounds = num_rounds
 
        # 环境内部的其他状态记录
        self.current_player = 0
        self.done = False
 
        # 初始化游戏需要在子类中实现
        #self.reset()
 
    def reset(self):
        # 初始化牌局，重新洗牌、发牌等
        # 在此进行游戏状态的重置
        self.hand_checker = MahjongHandChecker()
        self.yama_generator = YamaGenerator()
        self.logger = MahjongLogger(self.yama_generator.seed_str)
        self.msg = ""
        self.done = False

        # 初始化局数、本场数、立直棒数、立直状态、同巡振听状态
        self.round = [-1,0] # 局数,本场数
        self.num_kyoutaku = 0 # 供托数
        self.num_riichi = 0 # 立直棒数

        # 初始分数（四人）为25000点
        self.scores = [250 for _ in range(self.num_players)]

        # 初始化下一局游戏
        self.reset_for_next_round()

        # 返回当前玩家的观测
        return self.get_observation(self.current_player)

    def reset_for_next_round(self, oya_continue=False):
        # 初始化牌局，重新洗牌、发牌等
        # 在此进行游戏状态的重置
        # 这里仅作示例，需要自己实现具体逻辑
        
        # 随机生成初始状态（仅演示）
        # 公共信息（出牌历史）
        # 私有信息 （手牌，最多为14张，最少为13张）
        # shape: (14 * num_players,)
        #self.done = False
        self.to_riichi = False
        self.to_open_dora = 0
        self.final_penalty = True # 罚符指示器
        self.round_extended = False # 是否进入了延长战

        # 生成牌山 (136张牌，从最后一张开始摸牌，前14张为岭上牌)
        self.deck, self.dice = self.yama_generator.generate()

        # 初始化局数、本场数、立直棒数、立直状态、同巡振听状态
        if oya_continue:
            self.round[1] += 1
        else:
            self.round[0] += 1
            self.round[1] = 0

        # 立直棒数清零
        self.num_kyoutaku += self.num_riichi
        self.num_riichi = 0

        # 重置状态
        self.riichi = [False for _ in range(self.num_players)] # 立直状态
        self.ippatsu = [False for _ in range(self.num_players)] # 一发状态
        self.furiten_0 = [False for _ in range(self.num_players)] # 舍张振听状态
        self.furiten_1 = [False for _ in range(self.num_players)] # 同巡振听状态

        # 初始分数
        self.score_deltas = [0 for _ in range(self.num_players)]

        # 安排座次
        self.oya = self.round[0] % self.num_players # 庄家
        self.current_player = self.round[0] % self.num_players # 从庄家开始

        # 将14张岭上牌移出牌山，前四张是开杠时要补摸的牌， 第五张是第一张宝牌指示牌，第六张是其对应的里宝牌，以此类推
        self.dead_wall = [self.deck.pop(0) for _ in range(14)]
        self.kan_tile = [self.dead_wall.pop(i) for i in [1,0,1,0]]

        # 发初始手牌
        self.discard_pile = np.zeros((4, 136), dtype=bool)
        self.melds = [[] for _ in range(self.num_players)]
        self.hands = [[] for _ in range(self.num_players)]
        for _ in range(3):
            for p in range(self.num_players):
                # 发4张牌, 发3轮
                self.hands[(p+self.oya)%self.num_players] += [self.deck.pop() for _ in range(4)]
        for p in range(self.num_players):
                # 发1张牌
                self.hands[(p+self.oya)%self.num_players].append(self.deck.pop())
        self.last_discarded_tile = -1
        self.agari = None
        self.first_turn = [True for _ in range(self.num_players)]
        self.daburu_riichi = [False for _ in range(self.num_players)]
        self.can_kaze4 = True
        self.kaze4_tile = [0, 0]
        self.ryuukyoku_type = ""

        # 翻开宝牌
        self.dora_indicator = [self.dead_wall[1]]
        self.ura_indicator = []

        # 存储所有需要响应的动作
        self.claims = []
        self.is_selecting_tiles_for_claim = False
        self.selected_tiles = []

        # 设置当前阶段
        self.phase = "draw"  # 从庄家开始摸牌

        # 储存手牌的状态, 玩家的向听数
        self.machii = [[] for _ in range(self.num_players)]
        self.shanten = [self.hand_checker.check_shanten(hand) for hand in self.hands]
        self.tenpai = [False]*self.num_players # 只在游戏结束时才使用听牌状态

        # 输出天凤格式的log
        self.logger.add_init(self.round, self.num_kyoutaku, self.dice, self.dora_indicator, self.scores, self.oya, self.hands)
    def step(self, action_grp):
        """
        处理当前玩家的动作，然后判断是否有其他玩家吃碰杠和的机会，
        最后确定下一个 current_player并返回新的状态。
        """
        info = {}
        action, confirm = self.action_map(action_grp)

        if self.done:
            # 如果已经结束，返回当前状态即可（或 raise）
            return self.get_observation(self.current_player), 0.0, True, {}

        # 1. 取出当前玩家
        player = self.current_player
        
        # 2. 执行动作
        reward = 0.0
        match self.phase:
            case "draw":
                self.draw_tile(player)

                # 如果不处于立直状态，则解除同巡振听状态
                if not self.riichi[player]:
                    self.furiten_1[player] = False

                # 检查是否可以暗杠，加杠，立直，自摸或者流局
                claimed = self.check_self_claim(player)
                if not claimed:
                    # 自己没有宣告，进入弃牌阶段
                    self.phase = "discard"

                    # 消除自己的一发状态
                    self.ippatsu[player] = False
                else:
                    self.phase = self.claims[0]["type"] # 进入询问阶段
                    self.current_player = self.claims[0]["who"]

            case "kan_draw":
                # 如果是明杠/加杠，则在弃牌成功或者摸下一张岭上牌之后翻开一张dora
                self.to_open_dora += 1

                # 摸岭上牌
                self.draw_tile_from_dead_wall(player)

                # 如果连续杠，则在摸岭上牌时翻开之前的宝牌
                if self.to_open_dora > 1:
                    self.dora_indicator.append(self.dead_wall[len(self.dora_indicator)*2+1])
                    self.to_open_dora -= 1
                    # 记录天凤格式log e.g., <DORA hai="79" />
                    self.logger.add_dora(self.dora_indicator[-1])
                
                # 如果不处于立直状态，则解除同巡振听状态
                if not self.riichi[player]:
                    self.furiten_1[player] = False

                # 检查是否可以暗杠，加杠，立直，自摸或者流局
                claimed = self.check_self_claim(player, is_rinshan=True)
                if not claimed:
                    # 自己没有宣告，进入弃牌阶段
                    self.phase = "discard"
                else:
                    self.phase = self.claims[0]["type"] # 进入询问阶段
                    self.current_player = self.claims[0]["who"]
            
            case "riichi":
                # 立直逻辑
                # 持有 1000 点或以上而且有下一巡摸牌时可宣告立直
                # 打立直宣言牌放铳的情况下不需付供托(立直不成立)
                claim = self.claims.pop(0)
                if confirm:
                    self.riichi[player] = True
                    # 进入立直宣告阶段
                    self.to_riichi = True
                    # 检查两立直
                    if self.first_turn[player]:
                        self.daburu_riichi[player] = True
                    # 进入弃牌阶段，立直后第一张牌切出后必须听牌，之后直至和牌为止不能更改手牌
                    # 实现方法为，将要弃的牌与最后一张牌调换位置，然后进入弃牌阶段
                    self.hands[player][-1], self.hands[player][action] = self.hands[player][action], self.hands[player][-1]
                    self.phase = "discard"
                    # 输出天凤格式的log. <REACH who="3" step="1"/>
                    self.logger.add_reach_declare(player)
                else:
                    # 玩家选择什么都不做且有其他询问时，继续询问
                    if self.claims:
                        self.phase = self.claims[0]["type"]
                        self.current_player = self.claims[0]["who"]
                    else:
                        # 进入弃牌阶段，打出宣言牌，打出宣言牌成功后摆放一根立直棒
                        self.phase = "discard"

            case "discard":
                # 打牌逻辑
                if action < len(self.hands[player]):
                    # 丢弃牌
                    assert len(self.hands[player]) in [2, 5, 8, 11, 14], "出牌：手牌数不对"
                    tile_to_discard = self.hands[player][action]
                    self.hands[player].remove(tile_to_discard)
                    self.last_discarded_tile = tile_to_discard
                    self.discard_pile[player, tile_to_discard] = True # 加入弃牌堆

                    # 输出天凤格式的log. e.g. <D122/>
                    self.logger.add_discard(player, tile_to_discard)

                    # 更新待牌状态
                    self.update_machii(player)

                    # 检查向听数，如果下降了就给予奖励
                    new_shanten = self.hand_checker.check_shanten(self.hands[player])
                    #if not self.riichi[player]:
                    #    if new_shanten <= 1:
                    #        if new_shanten < self.shanten[player]:
                    #            reward = 0.1
                    #        elif new_shanten > self.shanten[player]:
                    #            reward = -0.1
                    
                    # 如果无役听牌或者振听给与惩罚
                    #if self.furiten_0[player] or self.furiten_1[player]:
                    #    reward += -0.1

                    self.shanten[player] = new_shanten

                    # 检查第一打是否是风牌
                    if self.can_kaze4 and self.first_turn[player]:
                        if self.kaze4_tile == [0, 0] and tile_to_discard // 4 in [27,28,29,30]:
                            self.kaze4_tile = [tile_to_discard // 4, 1]
                        elif self.kaze4_tile[0] == tile_to_discard // 4:
                            self.kaze4_tile[1] += 1
                        else:
                            self.can_kaze4 = False
                    else:
                        self.can_kaze4 = False

                    # 打出手牌后第一巡役种不再成立（地和，双立直）
                    self.first_turn[player] = False

                    # 检查是否有人可以和、碰、杠、吃
                    # 这通常涉及一串优先级判断和玩家询问过程
                    claimed = self.check_others_claim(tile_to_discard, player)
                    if not claimed:
                        # 如果是立直宣言牌，则立直成功，放置立直棒并扣1000点
                        self.complete_riichi(player)
                        # 如果是摸岭上后的弃牌，则弃牌成立并翻开新dora
                        if self.to_open_dora:
                            self.dora_indicator.append(self.dead_wall[len(self.dora_indicator)*2+1])
                            self.to_open_dora -= 1
                            # 记录天凤格式log e.g., <DORA hai="79" />
                            self.logger.add_dora(self.dora_indicator[-1])
                        # 无人吃碰杠和而且牌山还有剩余，则下一玩家摸牌
                        self.current_player = (player + 1) % self.num_players
                        self.phase = "draw"
                    else:
                        if self.claims[0]["type"] != 'ron':
                            # 立直宣言牌的情况下，检查claims中如果没有荣和，则立直成功并放置立直棒， 如果有，则先处理荣和
                            self.complete_riichi(player)
                            # 如果是摸岭上后的弃牌，检查claims中如果没有荣和，则弃牌成立并翻开新dora， 如果有，则先处理荣和
                            if self.to_open_dora:
                                self.dora_indicator.append(self.dead_wall[len(self.dora_indicator)*2+1])
                                self.to_open_dora -= 1
                                # 记录天凤格式log e.g., <DORA hai="79" />
                                self.logger.add_dora(self.dora_indicator[-1])
                        self.phase = self.claims[0]["type"] # 进入吃碰杠和询问阶段
                        self.current_player = self.claims[0]["who"]
                else:
                    # 无效动作
                    reward = -0.1
                    self.done = True
        
            # 3. 检查是否和牌
            # 如果有人和了，则计算奖励
            case "tsumo":
                claim = self.claims.pop(0)

                if confirm:
                    # 如果摸到岭上牌以后自摸，则在此时翻开一张dora
                    if self.to_open_dora:
                        self.dora_indicator.append(self.dead_wall[len(self.dora_indicator)*2+1])
                        self.to_open_dora -= 1
                        # 记录天凤格式log e.g., <DORA hai="79" />
                        self.logger.add_dora(self.dora_indicator[-1])

                    # 翻开里宝牌
                    if self.riichi[player] and not self.ura_indicator:
                        for i, _ in enumerate(self.dora_indicator):
                            self.ura_indicator.append(self.dead_wall[i*2])

                    # 立直棒数清零
                    self.num_kyoutaku += self.num_riichi
                    self.num_riichi = 0
                    self.agari = self.agari_calculation(player, player, claim["config"])
                    self.num_kyoutaku = 0

                    self.phase = "score"
                    self.current_player = 0
                else:
                    # 玩家选择什么都不做且有其他询问时，继续询问
                    if self.claims:
                        self.phase = self.claims[0]["type"]
                        self.current_player = self.claims[0]["who"]
                    else:
                        # 进入弃牌阶段
                        self.phase = "discard"

            case "ron":
                claim = self.claims.pop(0)
                fromwho = claim["fromwho"]
                # 玩家选择和牌时，本局结束并计算奖励
                if confirm:
                    # 清空声明序列 (只允许一次和牌)
                    self.claims = []

                    # 荣和的是立直宣言牌的情况下，立直不成立
                    self.to_riichi = False

                    # 如果荣和是明杠/加杠摸岭上牌后打出的牌，则不翻开新dora
                    self.to_open_dora = 0

                    # 更新游戏状态并计算奖励
                    self.hands[player] += [claim['tile']]

                    # 立直棒数清零
                    self.num_kyoutaku += self.num_riichi
                    self.num_riichi = 0
                    self.agari = self.agari_calculation(player, fromwho, claim["config"])
                    self.num_kyoutaku = 0

                    self.phase = "score"
                    self.current_player = 0
                else:
                    # 进入同巡振听状态
                    self.furiten_1[player] = True

                    # 玩家选择不和且有其他询问时，继续询问
                    if self.claims:
                        if self.claims[0]["type"] != 'ron':
                            # 立直宣言牌的情况下，检查claims中如果没有荣和，则立直成功并放置立直棒， 如果有，则先处理荣和
                            self.complete_riichi(fromwho)
                            # 如果是摸岭上后的弃牌，检查claims中如果没有荣和，则弃牌成立并翻开新dora， 如果有，则先处理荣和
                            if self.to_open_dora:
                                self.dora_indicator.append(self.dead_wall[len(self.dora_indicator)*2+1])
                                self.to_open_dora -= 1
                                # 记录天凤格式log e.g., <DORA hai="79" />
                                self.logger.add_dora(self.dora_indicator[-1])
                        self.phase = self.claims[0]["type"]
                        self.current_player = self.claims[0]["who"]
                    else:
                        # 玩家选择不和且没有其他询问
                        # 立直宣言牌的情况下，则立直成功并放置立直棒
                        self.complete_riichi(fromwho)
                        # 如果是摸岭上后的弃牌，则弃牌成立并翻开新dora
                        if self.to_open_dora:
                            self.dora_indicator.append(self.dead_wall[len(self.dora_indicator)*2+1])
                            self.to_open_dora -= 1
                            # 记录天凤格式log e.g., <DORA hai="79" />
                            self.logger.add_dora(self.dora_indicator[-1])

                        # 下一个玩家继续摸牌
                        self.current_player = (fromwho + 1) % self.num_players
                        self.phase = "draw"

            case "ryuukyoku":
                '''
                "yao9"   -> 九種九牌 # TODO: 无实现
                "reach4" -> 四家立直
                "ron3"   -> 三家和了 # TODO: 无实现
                "kan4"   -> 四槓散了
                "kaze4"  -> 四風連打
                "nm"     -> 流局满贯 # TODO: 无实现
                ""       -> 荒牌流局
                '''
                self.ryuukyoku_type = ""
                is_yao9 = False
                if is_yao9:
                    if confirm:
                        self.ryuukyoku_type = "九種九牌"
                        self.phase = "score"
                        self.final_penalty = False # 不进行罚符
                        self.current_player = 0
                    else:
                        self.claims.pop(0)
                        # 玩家选择不流局且有其他询问时，继续询问
                        if self.claims:
                            self.phase = self.claims[0]["type"]
                            self.current_player = self.claims[0]["who"]
                        else:
                            # 九种九牌是在摸牌后，丢牌前检查，所以不会有其他玩家的询问
                            self.phase = "draw"
                else:
                    if self.can_kaze4 and self.kaze4_tile[1] == 4:
                        self.ryuukyoku_type = "四風連打"
                        self.final_penalty = False # 不进行罚符
                    elif self.num_riichi == 4:
                        self.ryuukyoku_type = "四家立直"
                        self.final_penalty = False # 不进行罚符
                    elif len(self.kan_tile) == 0:
                        self.ryuukyoku_type = "四槓散了"
                        self.final_penalty = False # 不进行罚符
                    elif len(self.deck) == 0:
                        self.ryuukyoku_type = "荒牌流局"
                        self.final_penalty = True # 进行罚符
                    else:
                        self.ryuukyoku_type = "无效状态"
                        self.final_penalty = False

                    self.phase = "score"
                    self.current_player = 0

            case "score":
                player = self.current_player

                if player == 0:
                    # 更新听牌状态 （流局罚符用）
                    self.tenpai = [len(self.machii[player]) > 0 for player in range(self.num_players)]

                # 计算每家分数变动
                if self.agari:
                    self.score_deltas[player] = self.agari["sc"][player]
                else:
                    if self.final_penalty:
                        # 计算流局罚符
                        num_tenpai = sum(self.tenpai)
                        if 0 < num_tenpai < 4:
                            if self.tenpai[player]:
                                self.score_deltas[player] = MahjongEnvBase.RYUUKYOKU_PENALTY // num_tenpai
                            else:
                                self.score_deltas[player] =-MahjongEnvBase.RYUUKYOKU_PENALTY // (4 - num_tenpai)

                # 计算奖励
                if self.agari:
                    reward = 1 if self.score_deltas[player] > 0 else \
                            -1 if self.score_deltas[player] < 0 else 0
                else:
                    reward = 0.01 if self.tenpai[player] else -0.01

                # 结束更新分数流程，决定是否开始下一局或者宣告游戏结束
                if player == self.num_players - 1:
                    wind =     "東南西北"
                    bon = "一二三四五六七八九十"
                    self.msg += f"{wind[(self.round[0]//4) % 4]}{bon[self.round[0]%4]}局{bon[self.round[1]]}本场 "
                    # 记录本局信息
                    if self.agari:
                        if self.agari["who"] == self.agari["fromwho"]:
                            self.msg += f'''{tiles_printout(self.dora_indicator)}{"|"*self.num_riichi} 玩家{self.agari["who"]}{"立直+" if self.riichi[self.agari["who"]] else ""}自摸: {" ".join([tiles_printout(m["m"]) for m in self.agari["m"]])}+{tiles_printout(self.agari["hai"])}<-{tile_printout(self.agari["machi"])} {self.agari["yaku"]}'''
                        else:
                            self.msg += f'''{tiles_printout(self.dora_indicator)}{"|"*self.num_riichi} 玩家{self.agari["who"]}{"立直+" if self.riichi[self.agari["who"]] else ""}荣和玩家{self.agari["fromwho"]}: {" ".join([tiles_printout(m["m"]) for m in self.agari["m"]])}+{tiles_printout(self.agari["hai"])}<-{tile_printout(self.agari["machi"])} {self.agari["yaku"]}'''
                        # 记录天凤格式log <AGARI ba="1,1" hai="1,3,5,6,8,11,16,17" m="20935,51242" machi="3" ten="30,11700,0" yaku="20,1,34,2,54,1" doraHai="91" who="2" fromWho="2" sc="230,-40,210,-40,220,130,330,-40" />
                        rt=self.logger.add_agari(self.round, \
                                            self.num_riichi, \
                                            self.agari["hai"], \
                                            self.agari["m"], \
                                            self.agari["machi"], \
                                            self.agari["ten"], \
                                            self.scores, \
                                            self.score_deltas, \
                                            self.agari["yaku_tenhou"], \
                                            self.dora_indicator, \
                                            self.agari["who"], \
                                            self.agari["fromwho"], \
                                            self.ura_indicator, \
                                            self.agari.get("yakuman_tenhou", None), \
                                            self.agari.get("pao", None), \
                                            )
                    else:
                        self.msg += f"流局 原因: {self.ryuukyoku_type} {tiles_printout(self.dora_indicator)}" + "|"*self.num_riichi + f"{['听牌' if t else '不听' for t in self.tenpai]}"
                        # 记录天凤格式log
                        rt = self.logger.add_ryuukyoku(self.round, self.num_riichi, self.scores, self.score_deltas, self.hands, self.tenpai, self.num_players)
                    self.msg += "\n"

                    # 更新分数
                    for player in range(self.num_players):
                        self.scores[player] += self.score_deltas[player]

                    # 是否结束游戏？
                    match self.can_continue():
                        case 'game_over':
                            self.phase = "game_over"

                            assert sum(self.scores) + (self.num_riichi + self.num_kyoutaku) * 10 == 1000, "总分数不等于100000"

                            # 更新顺位
                            rank = sorted(list(range(self.num_players)), key=lambda x: self.scores[x])

                            # 若还有剩余点棒和供托则加给第一名
                            if self.scores[rank[0]]>self.scores[rank[1]]:
                                self.scores[rank[0]] += (self.num_riichi + self.num_kyoutaku) * 10
                            elif self.scores[rank[1]]>self.scores[rank[2]]:
                                self.scores[rank[0]] += (self.num_riichi + self.num_kyoutaku) * 5
                                self.scores[rank[1]] += (self.num_riichi + self.num_kyoutaku) * 5
                            elif self.scores[rank[2]]>self.scores[rank[3]]:
                                self.scores[rank[0]] += (self.num_riichi + self.num_kyoutaku) * 4
                                self.scores[rank[1]] += (self.num_riichi + self.num_kyoutaku) * 3
                                self.scores[rank[2]] += (self.num_riichi + self.num_kyoutaku) * 3
                            else:
                                self.scores[rank[0]] += (self.num_riichi + self.num_kyoutaku) * 3
                                self.scores[rank[1]] += (self.num_riichi + self.num_kyoutaku) * 3
                                self.scores[rank[2]] += (self.num_riichi + self.num_kyoutaku) * 2
                                self.scores[rank[3]] += (self.num_riichi + self.num_kyoutaku) * 2
                            self.num_riichi = 0
                            self.num_kyoutaku = 0

                            # 顺位马
                            rank_bonus = [50, 14, -26, -38]
                            for p in range(self.num_players):
                                self.score_deltas[p] = rank_bonus[rank[p]]

                            # 记录天凤格式log
                            self.logger.add_owari(rt, self.scores, self.score_deltas)

                        case 'next_round':
                            self.reset_for_next_round()
                            
                        case 'renchan':
                            self.reset_for_next_round(oya_continue=True)

                else:
                    self.current_player = (player + 1) % self.num_players
            
            case "game_over":
                # 更新顺位
                rank = list(reversed(sorted(list(range(self.num_players)), key=lambda x: self.scores[x])))

                # 游戏结束
                info = {"rank": rank,
                        "scores": self.scores,
                        "msg": self.msg,
                        "log": str(self.logger)}
                self.done = True

            case "ankan"|"chakan":
                # 暗杠/加杠逻辑
                claim = self.claims[0]
                if self.is_selecting_tiles_for_claim:
                    # 如果是暗杠，则在此时翻开一张dora
                    if self.phase == "ankan":
                        self.dora_indicator.append(self.dead_wall[len(self.dora_indicator)*2+1])
                        self.to_open_dora -= 1
                        # 记录天凤格式log e.g., <DORA hai="79" />
                        self.logger.add_dora(self.dora_indicator[-1])

                    # 将选中的牌从自己的手牌中移除
                    for tile in self.selected_tiles:
                        self.hands[player].remove(tile)
                    
                    # 将丢弃的牌加入自己的副露(Melds)中,清空选中的牌
                    if self.phase == "chakan":
                        for m in self.melds[player]:
                            if m["type"] == "pon" and m["claimed_tile"]//4 == self.selected_tiles[0]//4:
                                m["type"] = "chakan"
                                m["m"].append(self.selected_tiles[0])
                                # 记录天凤格式log e.g., <N who="2" m="25611" />
                                self.logger.add_meld(player, m)
                                break
                    else:
                        new_meld = {"type":"kan", 
                                    "fromwho":claim["fromwho"],  "offset":self.get_distance(player, claim["fromwho"]),
                                    "m": [t for t in self.selected_tiles], 
                                    "claimed_tile": None,
                                    "opened": False}
                        self.melds[player].append(new_meld)
                        # 记录天凤格式log e.g., <N who="2" m="25611" />
                        self.logger.add_meld(player, new_meld)

                    self.selected_tiles = []

                    # 消除所有的一发状态
                    self.ippatsu = [False for _ in range(self.num_players)]

                    # 鸣牌后第一巡役种不再成立（地和，双立直）
                    self.first_turn = [False for _ in range(self.num_players)]

                    # 鸣牌后四风连打不再成立
                    self.can_kaze4 = False

                    # 更新待牌状态
                    self.update_machii(player)
                    
                    # 如果没有其他玩家的声明，或者其他玩家选择不和，则进入摸岭上牌阶段
                    self.claims = []
                    self.is_selecting_tiles_for_claim = False
                    self.phase = "kan_draw"
                elif confirm:
                    self.is_selecting_tiles_for_claim = True
                    # 选择手牌中要丢弃的牌
                    num_tiles_to_select = 4 if self.phase == "ankan" else 1
                    #print(self.phase, action, self.hands[player])
                    claim_tile = self.hands[player][action]
                    
                    # 暗杠时手牌中必然有所有四张牌，加杠时手牌中必然只有一张对应的牌
                    tile_to_claim = [x for x in self.hands[player] if x // 4 == claim_tile // 4]
                    assert len(tile_to_claim) == num_tiles_to_select, "选择的牌数不对"
                    self.selected_tiles=tile_to_claim

                    # 开杠时检查其他玩家是否可以抢杠和
                    claimed = self.check_others_claim_chankan(tile_to_claim, player)
                    if claimed:
                        self.phase = self.claims[0]["type"]
                        self.current_player = self.claims[0]["who"]
                else:
                    # 什么都不做
                    self.claims.pop(0)
                    # 玩家选择什么都不做且有其他询问时，继续询问
                    if self.claims:
                        self.phase = self.claims[0]["type"]
                        self.current_player = self.claims[0]["who"]
                    else:
                        # 进入弃牌阶段
                        self.phase = "discard"

            case "pon"|"kan"|"chi":
                # 碰/杠/吃逻辑
                # 需要把丢弃的牌加入自己手中，并增加副露等
                # ...
                # 碰、杠后，当前玩家可以继续打牌(有些规则里杠完再摸一张) 
                # 这里可视需要决定是否继续由同一个玩家行动，或下一玩家
                claim = self.claims[0]
                fromwho = claim["fromwho"]
                if self.is_selecting_tiles_for_claim:
                    # 需要选择牌(需要进行动作合法性检查)
                    # 碰/吃需要选择两张牌
                    # （杠）需要选择第三张牌
                    num_tiles_to_select = 3 if self.phase == "kan" else 2

                    if len(self.selected_tiles) < num_tiles_to_select:
                        tile_to_claim = self.hands[player][action]
                        self.selected_tiles.append(tile_to_claim)
                        self.hands[player].remove(tile_to_claim)
                    else:
                        # 选择完毕
                        self.is_selecting_tiles_for_claim = False

                        # 将丢弃的牌加入自己的副露(Melds)中,清空选中的牌
                        self.last_discarded_tile = -1
                        self.discard_pile[fromwho, claim["tile"]] = False # 移出弃牌堆
                        new_meld = {"type":claim["type"], 
                                    "fromwho":fromwho, "offset":self.get_distance(player, fromwho),
                                    "m": sorted([t for t in self.selected_tiles] + [claim["tile"]]), 
                                    "claimed_tile": claim["tile"],
                                    "opened": True}

                        self.melds[player].append(new_meld)
                        self.selected_tiles = []

                        # 更新待牌状态
                        self.update_machii(player)

                        # （吃碰）接下来进入弃牌阶段，切一张牌
                        # （杠）接下来进入摸牌阶段，摸一张牌
                        self.phase="kan_draw" if self.phase == "kan" else "discard"

                        # 清空声明序列
                        self.claims = []

                        # 消除所有的一发状态
                        self.ippatsu = [False for _ in range(self.num_players)]

                        # 吃碰杠时等于开始了新的一巡，可以解除自己的同巡振听状态
                        self.furiten_1[player] = False

                        # 鸣牌后第一巡役种不再成立（地和，双立直）
                        self.first_turn = [False for _ in range(self.num_players)]

                        # 鸣牌后四风连打不再成立
                        self.can_kaze4 = False

                        # 记录天凤格式log e.g., <N who="2" m="25611" />
                        self.logger.add_meld(player, new_meld)
                elif confirm:
                    self.is_selecting_tiles_for_claim = True
                else:
                    # 什么都不做
                    self.claims.pop(0)
                    # 玩家选择什么都不做且有其他询问时，继续询问
                    if self.claims:
                        self.phase = self.claims[0]["type"]
                        self.current_player = self.claims[0]["who"]
                    else:
                        # 玩家选择什么都不做且没有其他询问
                        # 下一个玩家继续摸牌
                        self.current_player = (fromwho + 1) % self.num_players
                        self.phase = "draw"

        # 5. 组装返回
        obs = self.get_observation(self.current_player)
        return obs, reward, self.done, info
    
    def can_continue(self):
        # 是否结束游戏？
        # 0 - 游戏结束, 1 - 本局结束, 2 - 庄家连庄
        # 如果末亲听牌或者和牌而成为一位，或者达到八连庄，则游戏结束 (TODO: 加入sudden death 和延长战：南入和西入)
        oya_tenpai_or_agari = self.tenpai[self.oya] or (self.agari and self.agari["who"]== self.oya)
        renchan = self.round[1] >= MahjongEnvBase.MAX_HONBA
        sudden_death = min(self.scores) < 0
        if sudden_death:
            return 'game_over'
        elif self.round[0] == self.num_rounds - 1:
            oya_rank_top = self.scores[self.oya] == max(self.scores)
            if (oya_rank_top and oya_tenpai_or_agari) or renchan:
                return 'game_over'
            else:
                if not renchan and oya_tenpai_or_agari:
                    return 'renchan'
                else:
                    return 'game_over'
        else:
            if not renchan and oya_tenpai_or_agari:
               return 'renchan'
            else:
                return 'next_round'

    def agari_calculation(self, who, fromwho, config={}):
        config.update({ 
            "dora_indicators": self.dora_indicator + self.ura_indicator,
            "kyoutaku_number": self.num_kyoutaku,
            "tsumi_number": self.round[1],
            })
        result = self.hand_checker.calculate_hand_value(self.hands[who], self.hands[who][-1], self.melds[who], config, raise_error=True)

        sc = [result["cost"]["total"] if p == who else \
            -result["cost"]["main"]-result["cost"]["main_bonus"] if p == self.oya else \
            -result["cost"]["additional"]-result["cost"]["additional_bonus"] for p in range(self.num_players)] \
                if config["is_tsumo"] else \
             [result["cost"]["total"] if p == who else \
            -result["cost"]["main"]-result["cost"]["main_bonus"] if p == fromwho else 0 for p in range(self.num_players)]

        # Scale down by 100
        sc = [s//100 for s in sc]

        agari = {"hai":self.hands[who], "m":self.melds[who],\
                                   "machi":self.hands[who][-1], \
                                    "ten":[result["fu"],result["cost"]["total"],result["cost"]["yaku_level"]], \
                                    "yaku":result["yaku"], \
                                    "yaku_tenhou":result["yaku_tenhou"], \
                                    "yakuman_tenhou":result["yakuman_tenhou"], \
                                    "doraHai": [], \
                                    "doraHaiUra": [], \
                                    "who":who, "fromwho":fromwho, \
                                    "sc":sc}
        return agari

    def draw_tile(self, player):
        """摸牌逻辑：从牌山顶部拿一张给玩家,这对应于数组的尾部。"""
        assert len(self.hands[player]) in [1, 4, 7, 10, 13], "摸牌：手牌数不对"
        tile = self.deck.pop()
        self.hands[player].append(tile)
        # 输出天凤格式的log. e.g. <T86/>
        self.logger.add_draw(player, tile)
        # 更新待牌状态
        self.update_machii(player)

    def draw_tile_from_dead_wall(self, player):
        """摸牌逻辑：开杠后从岭上牌拿一张给玩家，然后从海底补一张进王牌堆。"""
        assert len(self.hands[player]) in [1, 4, 7, 10, 13], "摸岭上牌：手牌数不对"
        assert len(self.kan_tile) > 0, "岭上牌已经用完了"
        tile = self.kan_tile.pop(0)
        self.hands[player].append(tile)
        self.dead_wall.append(self.deck.pop(0))
        # 输出天凤格式的log. e.g. <T86/>
        self.logger.add_draw(player, tile)
        # 更新待牌状态
        self.update_machii(player)

    def complete_riichi(self, player):
        if self.to_riichi:
            self.num_riichi += 1
            self.scores[player] -= MahjongEnvBase.KYOUTAKU
            self.to_riichi = False
            self.ippatsu[player] = True
            # 输出天凤格式的log. e.g. <REACH who="3" ten="250,250,250,240" step="2"/>
            self.logger.add_reach_accepted(player, self.scores)
    
    def get_observation(self, player):
        """根据当前玩家，返回相应的状态表示。"""
        return {}
    
    def action_masks(self) -> list[bool]:
        """返回当前玩家的动作掩码。"""
        return []
    
    def check_self_claim(self, player, is_rinshan=False):
        """
        检查自己是否要立直，和牌，暗杠，加杠，流局

        Args:
            player (int): 打出牌的玩家编号。

        Returns:
            action (dict): 包含响应动作的信息。
                - "type": 动作类型 ("riichi", "tsumo", "ankan", "chakan", "ryuukyoku")。
                - "player": 响应动作的玩家编号。
                - "tile": 引发动作的牌。
        """
        # 定义动作优先级：和 > 杠 > 碰 > 吃
        action_priority = ["tsumo", "ryuukyoku", "riichi", "ankan", "chakan"]

        # 存储所有需要响应的动作
        self.claims = []

        # 1. 检查暗杠
        if self.can_ankan(player):
            self.claims.append({"type": "ankan", "fromwho": player, "who": player})

        # 2. 检查加杠
        if self.can_chakan(player):
            self.claims.append({"type": "chakan", "fromwho": player, "who": player})

        # 3. 检查立直
        if self.can_riichi(player):
            self.claims.append({"type": "riichi", "fromwho": player, "who": player})

        # 4. 检查自摸
        config = [None,]
        if self.can_tsumo(player, is_rinshan=is_rinshan, config=config):
            self.claims.append({"type": "tsumo", "fromwho": player, "who": player, "config": config[0]})

        # 5. 检查流局 (在抓牌的时候只有九种九牌流局)
        #if self.is_yao9(player):
        #    self.claims.append({"type": "ryuukyoku", "fromwho": player, "who": player})

        # 如果没有任何响应动作，返回 pass
        if not self.claims:
            return False
        
        # 根据优先级和玩家位置（距离）排序
        self.claims.sort(
            key=lambda x: action_priority.index(x["type"])
            )
        
        return True
    
    def can_riichi(self, player):
        # 立直要求手牌听牌，门清，且分数大于1000点
        if  not self.melds[player] and \
            not self.riichi[player] and \
            self.scores[player] >= MahjongEnvBase.KYOUTAKU and \
            self.hand_checker.check_shanten(self.hands[player]) <= 0:
            return True
        return False

    def has_yaku(self, player):
        """简单检测该玩家是否有役。调用一个和牌判断函数。"""
        # 检查是否有役
        who = player
        config={ \
            "is_tsumo": False,
            "is_riichi": self.riichi[who],
            "player_wind": (who+4-self.oya) % 4,
            "round_wind": (self.round[0] // 4) % 4}
        for m in self.machii[who]:
            result = self.hand_checker.calculate_hand_value(self.hands[who] + [m*4], m*4, self.melds[who], config)
            if result["error"] == 'no_yaku':
                return False
        return True
    
    def can_tsumo(self, player, is_rinshan=False, config = [None,]):
        """简单检测该玩家是否和牌。调用一个和牌判断函数。"""
        # 检查牌型是否能和牌
        # 检查是否有役
        who = player
        config[0]={ \
            "is_tsumo": True,
            "is_riichi": self.riichi[who],
            "is_ippatsu": self.ippatsu[who], # 检查自摸是否为一发
            "is_rinshan": is_rinshan, # 检查自摸是否为岭上
            #"is_chankan": False,
            "is_haitei": len(self.deck) == 0 and not is_rinshan, # 简单检查自摸是否为海底 （TODO 增加可选古役 花天月地 一般不允许）
            #"is_houtei": False,
            "is_daburu_riichi": self.daburu_riichi[who], # 检查自摸是否为两立直
            #"is_nagashi_mangan": False, # 自摸不可能是流局满贯
            "is_tenhou": self.first_turn[who] and who == self.oya, # 检查自摸是否为天和
            #"is_renhou": False,
            "is_chiihou": self.first_turn[who] and who != self.oya, # 检查自摸是否为地和
            #"is_open_riichi": False,
            "player_wind": (who+4-self.oya) % 4,
            "round_wind": (self.round[0] // 4) % 4}
        result = self.hand_checker.calculate_hand_value(self.hands[who], self.hands[who][-1], self.melds[who], config[0])
        return False if result["error"] else True
    
    def can_ankan(self, player):        
        if len(self.kan_tile) > 0 and \
            len(self.deck) > 1 :
            # TODO: 立直后暗杠不能改变听牌
            if self.riichi[player]:
                return False
        
            # 暗杠要求手牌中有四张相同的牌
            hands_34 = [tile // 4 for tile in self.hands[player]]

            for _, count in Counter(hands_34).items():
                if count >= 4:
                    return True

        return False
    
    def can_chakan(self, player):
        # 开明杠/暗杠/加杠要求还有岭上牌可以补充，并且牌山中至少还有两张牌
        # 加杠要求手牌中有三张相同的牌，且已经碰了        
        if self.melds[player] and \
            len(self.kan_tile) > 0 and \
            len(self.deck) > 1 :
            hands_34 = [tile // 4 for tile in self.hands[player]]

            for m in self.melds[player]:
                if m["type"] == "pon":
                    tile = m["claimed_tile"] // 4
                    for _, t in enumerate(hands_34):
                        if t == tile:
                            return True
        return False

    def can_ryuukyoku(self, _player):
        '''
        检查几种流局状态
            -> 九種九牌
            -> 四家立直
            -> 三家和了
            -> 四槓散了
            -> 四風連打
            -> 荒牌流局
        '''
        # 九种九牌可以选择主动流局
        if self.can_kaze4 and self.kaze4_tile[1] == 4:
            return True
        elif self.num_riichi == 4:
            return True
        elif len(self.kan_tile) == 0:
            return True
        elif len(self.deck) == 0:
            return True
        else:
            return False
    
    def check_others_claim_chankan(self, tiles, player):
        """
        检查其他玩家是否可以抢杠和。

        Args:
            tiles (list): 打出的牌。
            player (int): 打出牌的玩家编号。

        Returns:
            action (dict): 包含响应动作的信息。
                - "type": 动作类型 ("ron")。
                - "player": 响应动作的玩家编号。
                - "tile": 引发动作的牌。
        """
        # 存储所有需要响应的动作
        claims = []

        for other_player in range(self.num_players):
            if other_player == player:
                continue  # 跳过出牌玩家

            # 1. 检查和牌
            config = [None,]
            if self.can_ron(other_player, tiles[0], is_ankan=True if len(tiles) == 4 else False, is_chankan=True, config=config):
                claims.append({"type": "ron", "fromwho": player, "who": other_player, "tile": tiles[0], "config": config[0]})

        # 如果没有任何响应动作，返回 pass
        if not claims:
            return False

        # 根据玩家位置（距离）排序
        claims.sort(
            key=lambda x: self.get_distance(player, x["who"])
            )
        
        # 更新声明序列
        self.claims = claims + self.claims
        
        return True
    
    def check_others_claim(self, tile, player):
        """
        检查其他玩家是否要吃、碰、杠、和。

        Args:
            tile (int): 打出的牌。
            player (int): 打出牌的玩家编号。

        Returns:
            action (dict): 包含响应动作的信息。
                - "type": 动作类型 ("chow", "pong", "kong", "ron", "pass")。
                - "player": 响应动作的玩家编号。
                - "tile": 引发动作的牌。
        """
        # 定义动作优先级：和 > 杠 > 碰 > 吃
        action_priority = ["ron", "ryuukyoku", "kan", "pon", "chi"]

        # 存储所有需要响应的动作
        self.claims = []

        for other_player in range(self.num_players):
            if other_player == player:
                continue  # 跳过出牌玩家

            # 1. 检查和牌
            config = [None,]
            if self.can_ron(other_player, tile, config=config):
                self.claims.append({"type": "ron", "fromwho": player, "who": other_player, "tile": tile, "config": config[0]})

            # 2. 检查流局
            if self.can_ryuukyoku(player):
                self.claims.append({"type": "ryuukyoku", "fromwho": player, "who": player})

            # 3. 检查杠牌
            if self.can_kan(other_player, tile):
                self.claims.append({"type": "kan", "fromwho": player, "who": other_player, "tile": tile})

            # 4. 检查碰牌
            if self.can_pon(other_player, tile):
                self.claims.append({"type": "pon", "fromwho": player, "who": other_player, "tile": tile})

            # 5. 检查吃牌（仅对下家）
            if self.is_next_player(player, other_player) and self.can_chi(other_player, tile):
                self.claims.append({"type": "chi", "fromwho": player, "who": other_player, "tile": tile})

        # 如果没有任何响应动作，返回 pass
        if not self.claims:
            return False

        # 根据优先级和玩家位置（距离）排序
        self.claims.sort(
            key=lambda x: (action_priority.index(x["type"]), self.get_distance(player, x["who"]))
            )
        
        return True
    
    def update_machii(self, player):
        """更新玩家的待牌状态。"""
        self.machii[player] = self.hand_checker.check_machii(self.hands[player], self.melds[player])

        # 检查是否舍张振听
        self.furiten_0[player] = False
        for t in self.machii[player]:
            if sum(self.discard_pile[player, (t//4)*4:(t//4+1)*4]) > 0:
                self.furiten_0[player] = True
                return
        
        # 即使牌河的牌没构成振听，被鸣走的牌也可以构成振听
        for melds in [melds for i, melds in enumerate(self.melds) if i != player]:
            for meld in melds:
                if meld["fromwho"] == player and meld["claimed_tile"]//4 in self.machii[player]:
                    self.furiten_0[player] = True
                    return
                
    def can_ron(self, player, tile, is_ankan=False, is_chankan=False, config = [None,]):
        """简单检测该玩家是否和牌。调用一个和牌判断函数。"""        
        # 检查振听状态
        if MahjongEnvBase.HAS_FURITEN:
            # 检查是否同巡振听
            if self.furiten_1[player]:
                return False
            
            # 检查是否舍张振听
            if self.furiten_0[player]:
                return False
        
        if is_ankan:
            # 处于暗杠期间，检查自己手牌是否为国士无双听牌
            return self.hand_checker.check_win_condition_kokushi(self.hands[player]+[tile])
        else:
            # 检查牌型是否能和牌(包含正常牌型，七对和国士)
            who = player
            config[0]={ \
                "is_tsumo": False,
                "is_riichi": self.riichi[who],
                "is_ippatsu": self.ippatsu[who], # 检查荣和是否为一发
                #"is_rinshan": False, # 荣和不可能是岭上
                "is_chankan": is_chankan, # 检查荣和是否为抢杠
                #"is_haitei": False, # 荣和不可能是海底
                "is_houtei": len(self.deck) == 0, # 检查荣和是否为河底
                "is_daburu_riichi": self.daburu_riichi[who], # 检查是否为两立直
                #"is_nagashi_mangan": False, # 荣和不可能是流局满贯
                #"is_tenhou": False, # 荣和不可能是天和
                #"is_renhou": False,
                #"is_chiihou": False, # 荣和不可能是地和
                #"is_open_riichi": False,
                "player_wind": (who+4-self.oya) % 4,
                "round_wind": (self.round[0] // 4) % 4}
            result = self.hand_checker.calculate_hand_value(self.hands[who]+[tile], tile, self.melds[who], config[0])
            if result["error"]:
                return False
        return True
    
    def can_kan(self, player, tile):
        # 开明杠/暗杠/加杠要求还有岭上牌可以补充，并且牌山中至少还有两张牌
        if MahjongEnvBase.HAS_NAKU and \
            len(self.kan_tile) > 0 and \
            len(self.deck) > 1 and \
            not self.riichi[player]:
            return any(self.get_claim_tile_mask(tile, [], self.hands[player], "kan"))
        return False

    def can_pon(self, player, tile):
        # 检查是否不处于立直状态，且有人打出了一张与自己手牌中的两张相同的牌
        if MahjongEnvBase.HAS_NAKU and len(self.deck) > 0 and not self.riichi[player]:
            return any(self.get_claim_tile_mask(tile, [], self.hands[player], "pon"))
        return False

    def can_chi(self, player, tile):
        # 检查是否不处于立直状态，且有人打出了一张可以吃的牌
        # 检查食替，当吃完只有与被吃牌相同的牌能在同一巡被打出时，不能吃
        if MahjongEnvBase.HAS_NAKU and len(self.deck) > 0 and not self.riichi[player]:
            return any(self.get_claim_tile_mask(tile, [], self.hands[player], "chi"))
        return False

    def is_next_player(self, current_player, other_player):
        """判断某玩家是否是当前玩家的下家。"""
        return (current_player + 1) % self.num_players == other_player

    def get_distance(self, current_player, other_player):
        """计算当前玩家到其他玩家的距离（逆时针）。"""
        return (other_player - current_player) % self.num_players
    
    def get_claim_tile_mask(
    self,
    claim_tile_136: int,
    selected_tiles_136: list[int],
    hand_136: list[int],
    phase: str,
    ) -> list[bool]:
        """
        根据 phase 判断当前操作类型(吃 chi / 碰 pon / 杠 kan)，
        并结合 claim_tile、selected_tiles、hand 返回一个和 hand 等长的布尔列表。
        每个元素表示对应位置的牌是否“可被选中”来完成吃/碰/杠。
        
        :param claim_tile: 当前被他人打出的牌(或摸到的牌)，需要用来吃/碰/杠的目标
        :param selected_tiles: 已经在选择中的手牌列表（通常是索引或牌值）
        :param hand: 当前玩家手牌的列表
        :param phase: 当前玩家的操作类型，"chi", "pon", "kan"
        :return: 与 hand 等长的布尔列表，True 表示该位置可选，False 表示不可选
        """
        mask = [False] * len(hand_136)
        claim_tile = claim_tile_136 // 4
        selected_tiles = [t // 4 for t in selected_tiles_136]
        hand = [t // 4 for t in hand_136]
        
        # 根据 phase 判断要完成的动作和所需牌数
        if phase == "pon":
            # 碰：需要在手里再拿 2 张与 claim_tile 相同的牌
            needed_count = 2
            # 已经选了多少张？
            already_selected_count = sum(1 for t in selected_tiles if t == claim_tile)
            
            # 若还需要的数量 > 0，则标记手牌中符合条件的牌为 True
            to_select = needed_count - already_selected_count
            if to_select > 0 and hand.count(claim_tile) >= to_select:
                for i, t in enumerate(hand):
                    # 如果这张牌与 claim_tile 相同，并且还没被选中到足够数量
                    if t == claim_tile:
                        # 无需判断这张牌是否还没在 selected_tiles 里，因为之前选中的牌会被移除出手牌加入 self.selected_tiles
                        mask[i] = True

        elif phase == "kan":
            # 明杠：需要在手里再拿 3 张与 claim_tile 相同的牌
            needed_count = 3
            # 已经选了多少张？
            already_selected_count = sum(1 for t in selected_tiles if t == claim_tile)
            
            to_select = needed_count - already_selected_count
            if to_select > 0 and hand.count(claim_tile) >= to_select:
                for i, t in enumerate(hand):
                    if t == claim_tile:
                        mask[i] = True

        elif phase == "chi":
            # 吃：需要再拿 2 张，与 claim_tile 组成一个三连（同花色，顺序相邻）
            # 1. 若 claim_tile >= 27（字牌），无法吃
            if claim_tile >= 27:
                # 全部 False
                # 将mask的长度补齐到14
                mask += [False] * (14 - len(mask))
                return mask

            suit = claim_tile // 9  # 0:万, 1:条, 2:筒
            rank = claim_tile % 9   # 0~8

            # 2. 枚举所有可能包含 claim_tile 的三连
            possible_sequences = []
            for start_rank in (rank - 2, rank - 1, rank):
                # start_rank 必须在 0~6 范围内才不会越界
                if 0 <= start_rank <= 6:
                    seq = [start_rank, start_rank + 1, start_rank + 2]
                    if rank in seq:
                        possible_sequences.append(seq)

            # 如果没有任何可能组合（比如 rank = 0 或 8 时有且只有一个组合），则直接返回
            if not possible_sequences:
                # 将mask的长度补齐到14
                mask += [False] * (14 - len(mask))
                return mask

            # 3. 计算每个可能三连还需要哪些牌
            '''
            #    因为存在多个三连都能用到同一张牌的情况，我们要做“并集”意义上的统计：
            #    只要某一三连对某张牌 still needs > 0，我们就应该把该张牌标记可选。
            #    具体做法：先把所有三连的需求算出来，再合并。
            #    （还有一种做法：先判断玩家已经选中的牌能满足哪个三连，再针对那个三连继续判断需要哪些牌。
            #      不过为了兼容可能让玩家先选 A、又想换 B 的情况，这里用并集处理是更灵活的。）
            '''
            # 统计“对每个可能的三连”，手里还缺哪些牌
            # sequences_needed[tile] = 在任意一个三连中还需要 tile 的数量
            # 为了简化，我们只统计 True/False，而不是精确到数量
            sequences_needed = Counter()

            # 因为玩家需要从手中再选 2 张牌，若 selected_tiles 已选 n 张，那么还差 (2-n) 张
            # 但具体到不同三连可能已经选了不一样的牌，所以这里要分别算
            for seq in possible_sequences:
                # 将三连转换为牌索引（同花色，rank 分别是 seq[0], seq[1], seq[2]）
                seq_tiles = [suit * 9 + r for r in seq]
                # 这里包含了 claim_tile，但 claim_tile 是别人打出来的，不需要玩家选
                # 所以真正需要玩家选的只有 seq_tiles 去掉 claim_tile 之后的那两张
                needed = Counter(seq_tiles)
                needed[claim_tile] -= 1  # 去掉要吃的那张
                if needed[claim_tile] <= 0:
                    del needed[claim_tile]
                # 此时 needed 就是这个三连需要从玩家手里拿到的 2 张牌（可能相同，若三连里有重复? 正常三连里不会重复）
                # 再把 already selected_tiles 里对应的牌减掉
                for st_tile in selected_tiles:
                    if st_tile in needed:
                        needed[st_tile] -= 1
                        if needed[st_tile] <= 0:
                            del needed[st_tile]
                    else:
                        # 如果 selected_tiles 里有一张牌不在 needed 里，说明不合法
                        needed = Counter()  # 清空
                        break
                # 现在 needed 里剩下的就是玩家还差的牌
                # 如果 needed 为空，说明玩家手里的 selected_tiles 已经够这个三连了
                # 如果 needed 不为空，则说明还需要至少 1 张或 2 张，但是注意所有的牌都必须存在于手牌中
                # 我们把 needed 中“仍然 >0 的牌”标记一下 sequences_needed[tile] = True
                for needed_tile, cnt in needed.items():
                    if cnt > 0:
                        if hand.count(needed_tile) < cnt:
                            needed = Counter()

                for needed_tile, cnt in needed.items():
                    if cnt > 0:
                        sequences_needed[needed_tile] = True

            # 4. 生成最后的布尔掩码
            #    只要某张手牌在 sequences_needed 里为 True，并且没有超量使用（即手牌里还剩可选数量），就标记 True
            #    这里“超量使用”的判定需要手牌中 tile 的总数 - 已选同 tile 数 > 0
            for i, t in enumerate(hand):
                # 如果 sequences_needed[t] 为 True，说明至少有一个三连还缺它
                if sequences_needed.get(t, False):
                    mask[i] = True
                    assert hand[i] < 27, f"hand[i]={hand[i]}"

        # 将mask的长度补齐到14
        mask += [False] * (14 - len(mask))

        return mask

    def tiles_136_to_bool(self, tiles_136):
        """将136张牌表示转换为bool表示。"""
        tiles_bool = [False]*136
        for tile in tiles_136:
            tiles_bool[tile] = True
        return tiles_bool

    def tiles_bool_to_34(self, tiles_bool):
        """将bool表示转换为34表示。"""
        return [sum(tiles_bool[i:i+4]) for i, _ in enumerate(tiles_bool) if i % 4 == 0]

    def tiles_bool_to_4x34(self, tiles_bool):
        """将bool表示转换为34表示。"""
        return np.array([[t for i, t in enumerate(tiles_bool) if i % 4 == 0],
                [t for i, t in enumerate(tiles_bool) if i % 4 == 1],
                [t for i, t in enumerate(tiles_bool) if i % 4 == 2],
                [t for i, t in enumerate(tiles_bool) if i % 4 == 3]])

    def tiles_136_to_4x34(self, tiles_136):
        """将136张牌表示转换为4x34表示。"""
        return self.tiles_bool_to_4x34(self.tiles_136_to_bool(tiles_136))
    
    def action_map(action_grp):
        return action_grp


class MahjongEnv(MahjongEnvBase):
    """
    一个简化的麻将环境示例。
    """
    def __init__(self, num_players=4, num_rounds=4):
        super(MahjongEnv, self).__init__(num_players=num_players, num_rounds=num_rounds)

        # 定义动作空间
        # 维度1: 0 ~ 13: 切出14张手牌中的哪一张
        # 维度2: 0, 1: 取消/确认
        self.action_space = spaces.MultiDiscrete([14, 2])
 
        # 定义状态空间 (仅示例，具体需要你根据状态表示来定)
        # 比如这里假设每个玩家手中最多 14 张牌，每张牌取值 0~33 (共 34 张类型)
        # 加上其他上下文信息(如场风、圈风、剩余牌数等)，也可以用多维向量或 Box 
        # 这里只是给出一个极简示例
        #self.observation_space = spaces.Box(
        #    low=0, high=135, shape=(14,), dtype=np.int32
        #)
        self.observation_space = spaces.Dict(
            {
                "player": spaces.Discrete(4),
                "round": spaces.MultiDiscrete([MahjongEnvBase.MAX_ROUND, MahjongEnvBase.MAX_HONBA]),
                "hands": spaces.Box(low=-1, high=135, shape=(14,), dtype=np.int32),
                "discard_pile": spaces.Box(low=0, high=4, shape=(34, 4), dtype=np.int32),
                "last_discarded_tile": spaces.Box(low=-1, high=135, shape=(1,), dtype=np.int32),
                "melds": spaces.Box(low=0, high=4, shape=(34, 4), dtype=np.int32),
                "dora_indicator": spaces.Box(low=-1, high=135, shape=(5,), dtype=np.int32),
                "phase": spaces.Discrete(12),  
            }
        )

        self.reset()

    def get_observation(self, player):
        """根据当前玩家，返回相应的状态表示。"""
        return self.logger.snapshot_before_discard(player), {'who':player}
    
    def action_masks(self) -> list[bool]:
        match self.phase:
            case "discard":
                if self.riichi[self.current_player]:
                    # 立直后只能打最后摸到的牌
                    return [i == len(self.hands[self.current_player]) - 1 for i in range(14)] + [False, False]
                else:
                    return [0 <= tile < 136 for tile in self.hands[self.current_player]] + \
                            [False]*(14 - len(self.hands[self.current_player])) + [False, False]

            case "pon"|"kan"|"chi":
                # 先判断是否正在“从手牌里选哪几张来组合吃/碰/杠”
                if self.is_selecting_tiles_for_claim:
                    # 这里的 selected_tiles 可能是已经在 UI 或者其它逻辑里选了一些牌的记录
                    possible_mask = self.get_claim_tile_mask(
                        self.claims[0]["tile"],
                        self.selected_tiles,
                        self.hands[self.current_player],
                        self.phase
                    )
                    assert len(possible_mask) == 14, f"len(possible_mask)={len(possible_mask)}"
                    return possible_mask + [False, False]
                else:
                    return [False] * 14 + [True, True]
            
            case "ron"|"tsumo"|"ryuukyoku":
                return [False] * 14 + [True, True]
            
            case "ankan":
                mask = [False] * 14
                hands_34 = [tile // 4 for tile in self.hands[self.current_player]]
 
                for tile, count in Counter(hands_34).items():
                    if count >= 4:
                        for i, t in enumerate(hands_34):
                            if t == tile:
                                mask[i] = True

                return mask + [True, True]

            case "chakan":
                mask = [False] * 14
                hands_34 = [tile // 4 for tile in self.hands[self.current_player]]
 
                for m in self.melds[self.current_player]:
                    if m["type"] == "pon":
                        tile = m["claimed_tile"] // 4
                        for i, t in enumerate(hands_34):
                            if t == tile:
                                mask[i] = True

                return mask + [True, True]
            
            case "riichi":
                # 立直时只能打能使手牌听牌的牌
                assert len(self.hands[self.current_player]) == 14, "立直时手牌必须是14张"
                shantens = self.hand_checker.check_shantens(self.hands[self.current_player])
                return [s <= 0 for s in shantens] + [True, True]

        return [False] * 16
    
    def action_map(self, action_grp):
        return action_grp


if __name__ == "__main__":
    # 创建环境
    env = MahjongEnv(num_players=1)

    obs = env.reset()
    while not env.done:
        obs, reward, done, info = env.step(0)
        print(tiles_printout(obs["hands"]), reward, done, info["msg"])

    print("----------")
    env = MahjongEnv(num_players=4)

    obs = env.reset()
    while not env.done:
        obs, reward, done, info = env.step(0)
        print(tiles_printout(obs["hands"]), reward, done, info["msg"])

    print("----------")
    env = MahjongEnv(num_players=1)
    
    obs = env.reset()
    env.hands[env.current_player] = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 33, 34]
    env.deck[-1] = 35
    while not env.done:
        obs, reward, done, info = env.step(140)
        print(tiles_printout(obs["hands"]), reward, done, info["msg"])