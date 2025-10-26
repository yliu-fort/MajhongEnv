# riichi_minimal_engine.py
from __future__ import annotations
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
from dataclasses import dataclass, field
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple
import time

from my_types import *
from abstract import ARoomEngine, ATransport
from client import Strategy

from mahjong_env import MahjongEnv
#from mahjong_features import get_action_type_from_index

# ---------- 等待工具 ----------

def wait_some(futures, timeout: float):
    """返回部分完成的 futures（非阻塞微等待），避免忙等。"""
    done = set()
    not_done = set()
    end = time.time() + timeout
    for f in futures:
        if f.done():
            done.add(f)
        else:
            not_done.add(f)
    if not done and timeout > 0:
        # 小睡一会儿，降低 CPU
        sleep_left = max(0.0, end - time.time())
        if sleep_left > 0:
            time.sleep(min(sleep_left, 0.01))
        for f in list(not_done):
            if f.done():
                done.add(f); not_done.remove(f)
    return done, not_done


# ---------- 房间引擎 ----------

class RoomEngine(ARoomEngine):
    def __init__(self, room_id: str, seats: List[Seat], transport: ATransport):
        super().__init__(room_id, seats, transport)
        self.env = MahjongEnv()
        self.obs = self.env.reset()
        self._fallback_strategy = Strategy()

    # ---- 收集回应（带超时与“提前截断”能力） ----
    def collect_responses(self, step_id: int, requests: Dict[Request]) -> List[Response]:
        collected: Dict[Response] = {}
        if requests is not None and requests != {}:
            futures: Dict[Seat, Future] = {}
            deadline = max(r.deadline_ms for r in requests.values())
            now_ms = lambda: int(time.monotonic_ns() // 1_000_000)
            # 提交
            for req in requests.values():
                if len(req.actions) > 1: # 仅当动作数大于1的时候向客户端发送询问
                    futures[req.to_seat] = self.transport.send_request(req)
            
            # 轮询等待直到：到齐 或 超时 或 收到不可被覆盖的最高优先级（RON)
            while now_ms() < deadline and futures:
                done, _ = wait_some(futures.values(), timeout=0.05)
                for fut in list(done):
                    # 找到对应 seat
                    seat = None
                    for s, f in list(futures.items()):
                        if f is fut:
                            seat = s
                            break
                    if seat is None:
                        continue
                    try:
                        resp = fut.result(timeout=0)
                        if resp.step_id == step_id:
                            collected[seat] = resp
                    except Exception as e:
                        # 异常视为 PASS（不追加）
                        print(e)
                        pass
                    finally:
                        futures.pop(seat, None)

                # 提前截断：一旦有人 RON，直接返回（最高优先）
                #if any(r.chosen.action_type == ActionType.RON for r in collected.values()):
                #    break

        # 若都没返回，继续 loop；超时后自动退出
        for _ in requests.keys():
            if _ not in collected.keys():
                #allowed_action_type = set([get_action_type_from_index(i) for i in requests[_].actions])
                #print(f"{_} 超时！请求为 {allowed_action_type}")
                collected[_] = Response(room_id=self.room_id, \
                                step_id=step_id, \
                                request_id=f"req-{step_id}-{_}", \
                                from_seat=_, \
                                chosen=self._fallback_strategy.choose(requests[_]))

        # 过滤没有动作和动作为pass的回应
        collected = [r for r in collected.values() if r.chosen is not None and r.chosen.action_type != ActionType.PASS]
        return collected

    # ---- 对外：一个 step 演示 ----
    def step(self, window_ms: int = 5000):
        with self.lock:
            self.step_id += 1
            step_id = self.step_id

        requests: Dict[Request] = {}
        deadline = int(time.monotonic_ns() // 1_000_000) + window_ms
        for _, obs in self.obs.items():
            actions = [i for i, flag in enumerate(obs.legal_actions_mask) if flag]
            if len(actions) > 0:
                req = Request(
                    room_id=self.room_id,
                    step_id=step_id,
                    request_id=f"req-{step_id}-{Seat(_)}",
                    to_seat=Seat(_),
                    actions=actions,
                    observation=obs,
                    deadline_ms=deadline
                )
                requests[Seat(_)] = req

        responses = self.collect_responses(step_id, requests)
        decision = self.env.arbitrate(responses)
        self.obs,_,done,_,info = self.env.apply_decision(decision)
        return done, info