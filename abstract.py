# abstract.py
from __future__ import annotations
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import List, Dict, Optional, Tuple
import threading
from my_types import Seat, ActionSketch, Request, Response

# ---------- 传输层抽象（未来可替换为 WebSocket） ----------

class ATransport(ABC):
    """抽象传输层。此最小实现通过线程池直接调用客户端的处理函数。"""
    @abstractmethod
    def send_request(self, req: Request) -> Future:
        raise NotImplementedError

# ---------- 客户端抽象 ----------

class AClient(ABC):
    """最简客户端：根据策略在动作集中做出选择。"""
    def __init__(self, seat: Seat, name: str, strategy: AStrategy):
        self.seat = seat
        self.name = name
        self.strategy = strategy

    @abstractmethod
    def handle_request(self, req: Request) -> Response:
        raise NotImplementedError

# ---------- 策略抽象 ----------

class AStrategy(ABC):
    @abstractmethod
    def choose(self, req: Request) -> ActionSketch:
        raise NotImplementedError


# ---------- 房间引擎抽象 ----------

class ARoomEngine(ABC):
    def __init__(self, room_id: str, seats: List[Seat], transport: ATransport):
        self.room_id = room_id
        self.seats = seats
        self.transport = transport
        self.step_id = 0
        self.lock = threading.RLock()

    @abstractmethod
    def collect_responses(self, step_id: int, requests: List[Request]) -> List[Response]:
        raise NotImplementedError

    @abstractmethod
    def step(self, window_ms: int = 2000):
        raise NotImplementedError


# ---------- 麻将引擎抽象 ----------

class AMahjongEnv(ABC):
    def __init__(self):
        # 初始化游戏需要在子类中实现
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        # 在此进行游戏状态的重置
        raise NotImplementedError

    @abstractmethod
    def reset_for_next_round(self, oya_continue: bool=False):
        # 初始化牌局，重新洗牌、发牌等
        raise NotImplementedError

    @abstractmethod
    def step(self, responses: Dict[int, Response]):
        raise NotImplementedError

    @abstractmethod
    def arbitrate(self, responses: List[Response]) -> Optional[List[Response]]:
        # 仲裁玩家动作
        raise NotImplementedError

    @abstractmethod
    def apply_decision(self, decisions):
        """
        处理当前玩家的动作，然后判断是否有其他玩家吃碰杠和的机会，
        最后确定下一个 current_player并返回新的状态。
        """
        raise NotImplementedError

    @abstractmethod
    def can_continue(self):
        """
        是否结束游戏? 0 - 游戏结束, 1 - 本局结束, 2 - 庄家连庄
        如果末亲听牌或者和牌而成为一位，或者达到八连庄，则游戏结束 (TODO: 加入sudden death 和延长战：南入和西入)
        """
        raise NotImplementedError

    @abstractmethod
    def agari_calculation(self, who, fromwho, claimed_tile, config={}):
        # 和牌函数接口
        raise NotImplementedError

    @abstractmethod
    def draw_tile(self, player):
        # 摸牌逻辑：从牌山顶部拿一张给玩家,这对应于数组的尾部。
        raise NotImplementedError

    @abstractmethod
    def draw_tile_from_dead_wall(self, player):
        # 摸牌逻辑：开杠后从岭上牌拿一张给玩家，然后从海底补一张进王牌堆。
        raise NotImplementedError

    @abstractmethod
    def complete_riichi(self, player):
        # 如果状态合法则完成当前立直
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, who, legal_actions):
        # 根据当前玩家，返回相应的状态表示。
        raise NotImplementedError

    @abstractmethod
    def action_masks(self, player) -> list[bool]:
        # 返回当前玩家的动作掩码。
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def is_yao9(self, player):
        # 检测该玩家手牌中幺九牌是否大于等于九张
        raise NotImplementedError

    @abstractmethod
    def can_riichi(self, player):
        # 立直要求手牌听牌，门清，且分数大于1000点
        raise NotImplementedError

    @abstractmethod
    def can_tsumo(self, player, is_rinshan=False, config = [None,]):
        # 检测该玩家是否和牌。调用一个和牌判断函数。
        raise NotImplementedError

    @abstractmethod
    def can_ankan(self, player):
        '''
        开明杠/暗杠/加杠要求还有岭上牌可以补充，并且牌山中至少还有两张牌
        TODO: 立直后暗杠不能改变听牌
        暗杠要求手牌中有四张相同的牌
        '''
        raise NotImplementedError

    @abstractmethod
    def can_chakan(self, player):
        # 开明杠/暗杠/加杠要求还有岭上牌可以补充，并且牌山中至少还有两张牌
        # 加杠要求手牌中有三张相同的牌，且已经碰了        
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    def update_machii(self, player):
        # 更新玩家的待牌状态。
        raise NotImplementedError

    @abstractmethod
    def can_ron(self, player, tile, is_ankan=False, is_chankan=False, config = [None,]):
        # 检测该玩家是否和牌。调用一个和牌判断函数。    
        raise NotImplementedError

    @abstractmethod
    def can_kan(self, player, tile):
        # 开明杠/暗杠/加杠要求还有岭上牌可以补充，并且牌山中至少还有两张牌
        raise NotImplementedError

    @abstractmethod
    def can_pon(self, player, tile):
        # 检查是否不处于立直状态，且有人打出了一张与自己手牌中的两张相同的牌
        raise NotImplementedError

    @abstractmethod
    def can_chi(self, player, tile):
        # 检查是否不处于立直状态，且有人打出了一张可以吃的牌
        # TODO: 检查食替，当吃完只有与被吃牌相同的牌能在同一巡被打出时，不能吃
        raise NotImplementedError

    @abstractmethod
    def is_next_player(self, current_player, other_player):
        # 判断某玩家是否是当前玩家的下家。
        raise NotImplementedError

    @abstractmethod
    def get_distance(self, current_player, other_player):
        # 计算当前玩家到其他玩家的距离（逆时针）。
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def tiles_136_to_bool(self, tiles_136):
        """将136张牌表示转换为bool表示。"""
        raise NotImplementedError

    @abstractmethod
    def tiles_bool_to_34(self, tiles_bool):
        """将bool表示转换为34表示。"""
        raise NotImplementedError

    @abstractmethod
    def tiles_bool_to_4x34(self, tiles_bool):
        """将bool表示转换为34表示。"""
        raise NotImplementedError

    @abstractmethod
    def tiles_136_to_4x34(self, tiles_136):
        """将136张牌表示转换为4x34表示。"""
        raise NotImplementedError

    @abstractmethod
    def action_map(self, player, action_grp):
        raise NotImplementedError

    @abstractmethod
    def compute_legal_actions(self) -> list[list[bool]]:
        raise NotImplementedError

    @abstractmethod
    def _compute_legal_actions_per(self, who) -> list[bool]:
        raise NotImplementedError