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
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.screenmanager import Screen, ScreenManager

from agent.human_player_agent import HumanPlayerAgent
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
        self._request_queue: "queue.Queue[Optional[Tuple[int, Any, Any, float]]]" = queue.Queue()
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
        masks: Any,
        deadline: float,
    ) -> None:
        if self._stop_event.is_set():
            return
        payload = (request_id, observation, masks, deadline)
        self._request_queue.put(payload)

    def poll(self) -> Optional[Tuple[int, Optional[int]]]:
        try:
            return self._response_queue.get_nowait()
        except queue.Empty:
            return None

    def flush(self) -> None:
        if isinstance(self.agent, HumanPlayerAgent):
            self.agent.cancel_turn()
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
            request_id, observation, masks, deadline = payload
            action: Optional[int] = None
            if isinstance(self.agent, HumanPlayerAgent):
                timeout = max(0.0, deadline - time.monotonic())
                try:
                    self.agent.begin_turn(observation, masks, timeout)
                    action = self.agent.wait_for_action()
                except TimeoutError:
                    action = None
                except Exception:
                    action = None
            elif self.agent is not None:
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


class StartMenuScreen(Screen):
    background_source = StringProperty("assets/texture/main_menu.png")


class MahjongKivyApp(App):
    """Entry point for the Kivy Mahjong visualiser."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._controllers: list[AgentController] = []
        self._agents: list[Any] = []
        self._fallback_agent: Optional[RandomDiscardAgent] = None
        self._pending_requests: dict[int, _PendingRequest] = {}
        self._request_ids = count()
        self._action_timeout = 300.0
        self._observation: Any = None
        self.env: Optional[MahjongEnv] = None
        self.wrapper: Optional[MahjongEnvKivyWrapper] = None
        self._screen_manager: Optional[ScreenManager] = None
        self._game_screen: Optional[Screen] = None
        self._drive_event = None

    def build(self):
        base_path = Path(__file__).resolve().parent
        layout_path = base_path / "assets" / "layout" / "mahjong_gui.kv"
        if layout_path.exists():
            Builder.load_file(str(layout_path))

        self._screen_manager = ScreenManager()
        start_menu = Factory.StartMenuScreen(name="start_menu")
        self._game_screen = Screen(name="game")
        self._screen_manager.add_widget(start_menu)
        self._screen_manager.add_widget(self._game_screen)

        return self._screen_manager

    def _drive_environment(self, _dt: float) -> None:
        if self.env is None or self.wrapper is None or not self._controllers:
            return

        result = self.wrapper.fetch_step_result()
        if result is not None:
            self._observation, _, done, _ = result
            if done:
                self.return_to_menu()
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
            masks = self.wrapper.action_masks()
            controller.submit(request_id, self._observation, masks, deadline)
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
            self._queue_action_and_clear(current_seat, controller, None)

    def on_stop(self) -> None:
        self._cleanup_game()
        super().on_stop()

    def _initialise_agents(self, human_seats: Sequence[int]) -> None:
        if self.env is None or self.wrapper is None:
            return

        num_players = self.env.num_players
        self._agents = [None] * num_players
        self._fallback_agent = RandomDiscardAgent(env=self.env)

        human_seat_set = set(human_seats)
        model_path = Path("model_weights/latest.pt")
        for seat in range(num_players):
            if seat in human_seat_set:
                agent = HumanPlayerAgent()
                self.wrapper.bind_human_ui(seat, agent)
                assist_agent: Optional[VisualAgent] = None
                try:
                    assist_agent = VisualAgent(self.env, backbone="resnet50")
                    if model_path.exists():
                        assist_agent.load_model(str(model_path))
                except Exception:
                    assist_agent = None
                if assist_agent is not None and self.wrapper is not None:
                    self.wrapper.set_assist_agent(seat, assist_agent)
            else:
                agent = VisualAgent(self.env, backbone="resnet50")
                agent.load_model("model_weights/latest.pt")
            if self.wrapper is not None:
                self.wrapper.register_seat_agent(seat, agent)
            self._agents[seat] = agent

    def _initialise_controllers(self) -> None:
        if self.env is None:
            return
        self._controllers = [
            AgentController(seat=index, agent=self._agents[index])
            for index in range(self.env.num_players)
        ]

    def _start_controllers(self) -> None:
        for controller in self._controllers:
            controller.start()

    def _start_game(self, human_seats: Sequence[int]) -> None:
        self._cleanup_game()

        self.env = MahjongEnv(num_players=4)
        self.wrapper = MahjongEnvKivyWrapper(env=self.env)
        self._pending_requests = {}
        self._request_ids = count()

        if self._game_screen is not None:
            self._game_screen.clear_widgets()
            self._game_screen.add_widget(self.wrapper.root)

        self._initialise_agents(human_seats)
        self._initialise_controllers()
        self._observation = self.wrapper.reset()
        self._start_controllers()

        interval = 1.0 / max(1, self.wrapper.fps)
        self._drive_event = Clock.schedule_interval(self._drive_environment, interval)

        if self._screen_manager is not None:
            self._screen_manager.current = "game"

    def return_to_menu(self, *args: Any) -> None:
        self._cleanup_game()

    def start_ai_vs_human(self) -> None:
        self._start_game(human_seats=(0,))

    def start_ai_vs_ai(self) -> None:
        self._start_game(human_seats=())

    def _queue_action_and_clear(
        self, seat: int, controller: AgentController, action: Optional[int]
    ) -> None:
        controller.flush()
        self._pending_requests.pop(seat, None)
        if action is None and self._fallback_agent is not None:
            action = self._fallback_agent.predict(self._observation)
        if action is None:
            return
        if self.wrapper is not None:
            self.wrapper.queue_action(action)

    def _handle_environment_reset(self) -> None:
        self.return_to_menu()

    def _flush_pending_requests(self) -> None:
        self._pending_requests.clear()
        for controller in self._controllers:
            controller.flush()

    def _shutdown_controllers(self) -> None:
        for controller in self._controllers:
            controller.stop()
        self._controllers = []

    def _cleanup_game(self, *args: Any) -> None:
        if self._drive_event is not None:
            self._drive_event.cancel()
            self._drive_event = None
        self._flush_pending_requests()
        self._shutdown_controllers()
        self._agents = []
        self._fallback_agent = None
        if self.wrapper is not None:
            self.wrapper.close()
            self.wrapper = None
        self.env = None
        self._observation = None
        self._pending_requests = {}
        self._request_ids = count()
        if self._game_screen is not None:
            self._game_screen.clear_widgets()
        if self._screen_manager is not None:
            self._screen_manager.current = "start_menu"


if __name__ == "__main__":
    MahjongKivyApp().run()
