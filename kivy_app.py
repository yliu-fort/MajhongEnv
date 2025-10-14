import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "agent"))

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder

from agent.visual_agent import VisualAgent
from agent.random_discard_agent import RandomDiscardAgent
from mahjong_env import MahjongEnv
from mahjong_wrapper_kivy import MahjongEnvKivyWrapper


@dataclass
class _PendingRequest:
    request_id: int
    deadline: float


class AgentController:
    """Background worker that evaluates agent policies on demand."""

    def __init__(self, seat: int, agent: Optional[Any]) -> None:
        self.seat = seat
        self.agent = agent
        self._request_queue: "queue.Queue[Optional[Tuple[int, Any, float]]]" = queue.Queue()
        self._response_queue: "queue.Queue[Tuple[int, Optional[int]]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"AgentController[{seat}]",
            daemon=True,
        )

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        self._request_queue.put(None)
        self._thread.join(timeout=timeout)
        self.flush()

    def submit(
        self,
        request_id: int,
        observation: Any,
        deadline: float,
    ) -> None:
        if self._stop_event.is_set():
            return
        payload = (request_id, observation, deadline)
        self._request_queue.put(payload)

    def poll(self) -> Optional[Tuple[int, Optional[int]]]:
        try:
            return self._response_queue.get_nowait()
        except queue.Empty:
            return None

    def flush(self) -> None:
        self._drain_queue(self._request_queue)
        self._drain_queue(self._response_queue)

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                continue
            request_id, observation, deadline = payload
            action: Optional[int] = None
            if self.agent is not None:
                try:
                    action = self.agent.predict(observation)
                except Exception:
                    action = None
            self._response_queue.put((request_id, action))

    @staticmethod
    def _drain_queue(q: "queue.Queue[Any]") -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break


class MahjongKivyApp(App):
    """Entry point for the Kivy Mahjong visualiser."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._controllers: list[AgentController] = []
        self._agents: list[VisualAgent] = []
        self._fallback_agent: RandomDiscardAgent = None
        self._pending_requests: dict[int, _PendingRequest] = {}
        self._request_ids = count()
        self._action_timeout = 1.0
        self._observation: Any = None

    def build(self):
        base_path = Path(__file__).resolve().parent
        layout_path = base_path / "assets" / "layout" / "mahjong_gui.kv"
        if layout_path.exists():
            Builder.load_file(str(layout_path))

        self.env = MahjongEnv(num_players=4)
        self.wrapper = MahjongEnvKivyWrapper(env=self.env)
        self._initialise_agents()
        self._initialise_controllers()
        self._observation = self.wrapper.reset()
        self._start_controllers()

        interval = 1.0 / max(1, self.wrapper.fps)
        Clock.schedule_interval(self._drive_environment, interval)
        return self.wrapper.root

    def _drive_environment(self, _dt: float) -> None:
        result = self.wrapper.fetch_step_result()
        if result is not None:
            self._observation, _, done, _ = result
            if done:
                self._flush_pending_requests()
                return

        if self.env.done and self.wrapper.pending_action is None:
            self._handle_environment_reset()
            return

        if self.wrapper.pending_action is not None:
            return

        current_seat = getattr(self.env, "current_player", 0)
        controller = self._controllers[current_seat]
        pending = self._pending_requests.get(current_seat)
        now = time.monotonic()

        if pending is None:
            deadline = now + self._action_timeout
            request_id = next(self._request_ids)
            controller.submit(request_id, self._observation, deadline)
            self._pending_requests[current_seat] = _PendingRequest(
                request_id=request_id,
                deadline=deadline,
            )
            return

        while True:
            response = controller.poll()
            if response is None:
                break
            request_id, action = response
            if request_id != pending.request_id:
                continue
            self._queue_action_and_clear(current_seat, controller, action)
            return

        if now >= pending.deadline:
            fallback_action = self._fallback_agent(self._observation)
            self._queue_action_and_clear(current_seat, controller, fallback_action)

    def on_stop(self) -> None:
        self._shutdown_controllers()
        if hasattr(self, "wrapper"):
            self.wrapper.close()
        super().on_stop()

    def _initialise_agents(self) -> None:
        base_agent = VisualAgent(self.env, backbone="resnet50")
        base_agent.load_model("model_weights/latest.pt")
        self._agents = [base_agent]
        for _ in range(1, self.env.num_players):
            agent = VisualAgent(self.env, backbone="resnet50")
            agent.model.load_state_dict(base_agent.model.state_dict())
            self._agents.append(agent)
        self._fallback_agent = RandomDiscardAgent(env=self.env)

    def _initialise_controllers(self) -> None:
        self._controllers = [
            AgentController(seat=index, agent=self._agents[index])
            for index in range(self.env.num_players)
        ]

    def _start_controllers(self) -> None:
        for controller in self._controllers:
            controller.start()

    def _queue_action_and_clear(
        self, seat: int, controller: AgentController, action: int
    ) -> None:
        self.wrapper.queue_action(action)
        controller.flush()
        self._pending_requests.pop(seat, None)

    def _handle_environment_reset(self) -> None:
        self._flush_pending_requests()
        self._observation = self.wrapper.reset()

    def _flush_pending_requests(self) -> None:
        self._pending_requests.clear()
        for controller in self._controllers:
            controller.flush()

    def _shutdown_controllers(self) -> None:
        for controller in self._controllers:
            controller.stop()


if __name__ == "__main__":
    MahjongKivyApp().run()
