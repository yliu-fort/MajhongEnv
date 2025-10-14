import os
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "agent"))

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder

from agent.human_player_agent import HumanPlayerAgent
from agent.random_discard_agent import RandomDiscardAgent
from mahjong_env import MahjongEnv
from mahjong_wrapper_kivy import MahjongEnvKivyWrapper


@dataclass
class _QueuedAction:
    player: int
    action: int


class _PlayerWorker(threading.Thread):
    """Background worker responsible for producing a player's actions."""

    def __init__(self, player_index: int, agent: Any, action_queue: queue.Queue):
        super().__init__(daemon=True)
        self.player_index = player_index
        self.agent = agent
        self._action_queue = action_queue
        self._requests: "queue.Queue[Optional[tuple[int, Any, Optional[Sequence[int]]]]]" = queue.Queue()
        self._stop_event = threading.Event()

    def request_action(
        self,
        request_id: int,
        observation: Any,
        legal_mask: Optional[Sequence[int]],
    ) -> None:
        self._requests.put((request_id, observation, legal_mask))

    def clear(self) -> None:
        while True:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                break
        if hasattr(self.agent, "cancel_pending"):
            self.agent.cancel_pending()

    def stop(self) -> None:
        self._stop_event.set()
        self._requests.put(None)
        if hasattr(self.agent, "cancel_pending"):
            self.agent.cancel_pending()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                break
            request_id, observation, legal_mask = payload
            if hasattr(self.agent, "set_player_index"):
                self.agent.set_player_index(self.player_index)
            if hasattr(self.agent, "set_observation"):
                self.agent.set_observation(observation)
            if hasattr(self.agent, "set_legal_actions"):
                mask_list: Optional[List[int]]
                if legal_mask is None:
                    mask_list = None
                else:
                    mask_list = list(legal_mask)
                self.agent.set_legal_actions(mask_list)
            action = self.agent.predict(observation)
            self._action_queue.put((self.player_index, action, request_id))


class _PlayerThreadManager:
    """Coordinates background threads for each active player."""

    def __init__(self, agents: Sequence[Any]):
        self._action_queue: "queue.Queue[tuple[int, int, int]]" = queue.Queue()
        self._workers = [
            _PlayerWorker(index, agent, self._action_queue)
            for index, agent in enumerate(agents)
        ]
        self._active_requests: List[Optional[int]] = [None] * len(self._workers)
        self._next_request_id = 1
        for worker in self._workers:
            worker.start()

    def request_action(
        self,
        player_index: int,
        observation: Any,
        legal_mask: Optional[Sequence[int]],
    ) -> None:
        if not (0 <= player_index < len(self._workers)):
            return
        if self._active_requests[player_index] is not None:
            return
        request_id = self._next_request_id
        self._next_request_id += 1
        self._active_requests[player_index] = request_id
        self._workers[player_index].request_action(request_id, observation, legal_mask)

    def poll_action(self) -> Optional[_QueuedAction]:
        while True:
            try:
                player, action, request_id = self._action_queue.get_nowait()
            except queue.Empty:
                return None
            expected = self._active_requests[player]
            if expected != request_id:
                continue
            self._active_requests[player] = None
            return _QueuedAction(player=player, action=action)

    def reset(self) -> None:
        for player, worker in enumerate(self._workers):
            worker.clear()
            self._active_requests[player] = None
        while True:
            try:
                self._action_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        for worker in self._workers:
            worker.stop()
        for worker in self._workers:
            worker.join(timeout=1.0)
        self.reset()


class MahjongKivyApp(App):
    """Entry point for the Kivy Mahjong visualiser."""

    def build(self):
        base_path = Path(__file__).resolve().parent
        layout_path = base_path / "assets" / "layout" / "mahjong_gui.kv"
        if layout_path.exists():
            Builder.load_file(str(layout_path))

        self.env = MahjongEnv(num_players=4)
        self.wrapper = MahjongEnvKivyWrapper(env=self.env)
        self._agents = self._build_agents()
        self._player_manager = _PlayerThreadManager(self._agents)
        self._observation = self.wrapper.reset()
        self._player_manager.reset()

        interval = 1.0 / max(1, self.wrapper.fps)
        Clock.schedule_interval(self._drive_environment, interval)
        return self.wrapper.root

    def _drive_environment(self, _dt: float) -> None:
        result = self.wrapper.fetch_step_result()
        if result is not None:
            self._observation, _, done, _ = result
            if done:
                return

        if self.env.done and self.wrapper.pending_action is None:
            self._player_manager.reset()
            if hasattr(self.wrapper, "dismiss_human_action"):
                self.wrapper.dismiss_human_action()
            self._observation = self.wrapper.reset()
            return

        if self.wrapper.pending_action is not None:
            return

        if not self.env.done:
            legal_mask = self.wrapper.action_masks()
            self._player_manager.request_action(
                self.env.current_player, self._observation, legal_mask
            )

        queued = self._player_manager.poll_action()
        if queued is not None and queued.player == self.env.current_player:
            if hasattr(self.wrapper, "dismiss_human_action"):
                self.wrapper.dismiss_human_action()
            self.wrapper.queue_action(queued.action)

    def on_stop(self) -> None:
        self._player_manager.stop()
        return super().on_stop()

    def _build_agents(self) -> Sequence[Any]:
        agents: List[Any] = []
        for idx in range(self.env.num_players):
            if idx == 0:
                agents.append(HumanPlayerAgent(self.env, self.wrapper, player_index=idx))
            else:
                agents.append(RandomDiscardAgent(self.env))
        return agents


if __name__ == "__main__":
    MahjongKivyApp().run()
