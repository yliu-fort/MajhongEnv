# transport.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, Future, ProcessPoolExecutor
from my_types import *
from abstract import AClient, ATransport

# ---------- 传输层抽象（未来可替换为 WebSocket） ----------

class InProcessTransport(ATransport):
    def __init__(self, client_registry: Dict[Seat, AClient], pool: ThreadPoolExecutor):
        self.clients = client_registry
        self.pool = pool

    def send_request(self, req: Request) -> Future:
        client = self.clients[req.to_seat]
        # 通过线程池异步执行，模拟网络请求-响应
        return self.pool.submit(client.handle_request, req)


class ProcessPoolTransport(ATransport):
    def __init__(self, client_registry: Dict[Seat, AClient], pool: ProcessPoolExecutor):
        self.clients = client_registry
        self.pool = pool

    def send_request(self, req: Request) -> Future:
        client = self.clients[req.to_seat]
        # 通过线程池异步执行，模拟网络请求-响应
        return self.pool.submit(client.handle_request, req)