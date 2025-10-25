# demo.py
from __future__ import annotations
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import numpy as np

from my_types import Seat
from transport import InProcessTransport
from client import Client, CNNStrategy
from engine import RoomEngine

# ---------- 演示主程序 ----------
def demo():
    # 客户端与策略
    clients: Dict[Seat, Client] = {
        Seat.EAST:  Client(Seat.EAST,  "East",  CNNStrategy()), 
        Seat.SOUTH: Client(Seat.SOUTH, "South", CNNStrategy()),
        Seat.WEST:  Client(Seat.WEST,  "West",  CNNStrategy()), 
        Seat.NORTH: Client(Seat.NORTH, "North", CNNStrategy()),
    }

    with ThreadPoolExecutor(max_workers=4, initializer=CNNStrategy.init_model) as pool:
        transport = InProcessTransport(clients, pool)

        for ep in range(100):
            total_dscores = np.zeros(4, dtype=np.int32)
            engine = RoomEngine(room_id="room-1", seats=[Seat.EAST, Seat.SOUTH, Seat.WEST, Seat.NORTH], transport=transport)

            done = False
            while not done:
                # In 'step', the engine first determines the tasks required before processing any client queries.
                # It then computes and broadcasts the current state and available legal actions to all clients.
                # After collecting all client responses, the engine performs arbitration to decide on an action.
                # Once a decision is made, the engine applies the selected action and updates its internal state.
                done, info = engine.step()
            
            total_dscores += np.array(info["scores"]) - 250
            print(f"Episode {ep} - 分数板：{total_dscores}", info["scores"])
            print(info["msg"])
            with open(f'../log_analyser/paipu/evaluate_log_{ep}.mjlog', "w") as f:
                f.write(info["log"])
            

if __name__ == "__main__":
    demo()
