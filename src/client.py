# client.py
from __future__ import annotations
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agent"))
from my_types import *
from abstract import AClient, AStrategy
import time
import random

from agent.visual_agent import VisualAgent
from mahjong_features import get_action_type_from_index

# ---------- 客户端 ----------

class Client(AClient):
    """最简客户端：根据策略在动作集中做出选择。"""
    def __init__(self, seat: Seat, name: str, strategy: AStrategy):
        super().__init__(seat, name, strategy)

    def handle_request(self, req: Request) -> Response:
        # 模拟人类思考延迟（可按策略决定）
        think_ms = self.strategy.think_time_ms(req)
        time.sleep(think_ms / 1000.0)
        choice = self.strategy.choose(req)
        #print(f"[Client {self.name}] step={req.step_id} choose={choice.action_type.name}")
        return Response(
            room_id=req.room_id,
            step_id=req.step_id,
            request_id=req.request_id,
            from_seat=self.seat,
            chosen=choice,
        )

# ---------- 策略 ----------

class Strategy(AStrategy):
    def think_time_ms(self, req: Request) -> int:
        return 1  # 统一 1ms 思考

    def choose(self, req: Request) -> ActionSketch:
        # 默认策略
        if len(req.actions) == 1:
            best = req.actions[0]
        else:
            best = random.choice([a for a in req.actions if a < 34 or a > 251]) # 不进行除了切牌以外的任何操作
        return ActionSketch(action_type=get_action_type_from_index(best), payload={"action_id": best})

class RandomStrategy(AStrategy):
    def think_time_ms(self, req: Request) -> int:
        return 1  # 统一 1ms 思考

    def choose(self, req: Request) -> ActionSketch:
        # 随机策略
        best = random.choice(req.actions)
        return ActionSketch(action_type=get_action_type_from_index(best), payload={"action_id": best})


class CNNStrategy(AStrategy):
    # 全局（仅在子进程内持有）
    MODEL = None

    @staticmethod
    def init_model():
        """在每个子进程启动时加载一次模型并搬到 MPS/CPU。"""
        if CNNStrategy.MODEL is None:
            model = VisualAgent(None, backbone="resnet50")
            model.load_model("model_weights/latest.pt")
            CNNStrategy.MODEL = model

    def __init__(self):
        super().__init__()
        
    def think_time_ms(self, req: Request) -> int:
        return 1  # 统一 1ms 思考

    def choose(self, req: Request) -> ActionSketch:
        if CNNStrategy.MODEL is None:
            raise NotImplementedError
        # resnet策略
        return CNNStrategy.MODEL.predict(req.observation)