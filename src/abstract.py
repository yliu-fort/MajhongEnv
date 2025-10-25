# abstract.py
from __future__ import annotations
from abc import ABC, abstractmethod
from concurrent.futures import Future
from my_types import *
import threading


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
