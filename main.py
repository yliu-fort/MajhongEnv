import os
import queue
import sys
import threading
import time
import random
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "agent"))

# 1) 彻底关闭 Kivy 的参数解析
os.environ["KIVY_NO_ARGS"] = "1"   # 必须在 import kivy 之前

from kivy.app import App
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.screenmanager import Screen, ScreenManager

from agent.human_player_agent import HumanPlayerAgent
from agent.visual_agent import VisualAgent as _AIAgent
#from agent.rule_based_agent import RuleBasedAgent as _AIAgent
from agent.random_discard_agent import RandomDiscardAgent
from mahjong_env import MahjongEnv
from mahjong_features import get_action_type_from_index
from mahjong_wrapper_kivy import MahjongEnvKivyWrapper
from my_types import ActionSketch, Response, Seat


@dataclass
class _PendingRequest:
    request_id: int
    deadline: float
    observation: Any


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
        self._step_index = 0
        self._room_id = "local"
        self.env: Optional[MahjongEnv] = None
        self.wrapper: Optional[MahjongEnvKivyWrapper] = None
        self._screen_manager: Optional[ScreenManager] = None
        self._game_screen: Optional[Screen] = None
        self._drive_event = None
        self._log_counts = 0

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
            (
                self._observation,
                rewards,
                terminations,
                truncations,
                info,
            ) = result
            done = self._is_done(terminations, truncations)
            if done:
                self.env.logger.write_to_file(f"output/{self._log_counts}.mjlog")
                self._log_counts += 1
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

        seat_observation = self._get_seat_observation(current_seat)
        mask = self._extract_mask(seat_observation)
        if mask is None:
            mask = self.wrapper.action_masks(current_seat)

        mask_options = self._mask_actionable(mask)
        if mask is not None and mask_options <= 1:
            self._queue_action_and_clear(
                current_seat,
                controller,
                self._auto_action_from_mask(mask),
                observation=seat_observation,
            )
            return

        if pending is None:
            deadline = now + self._action_timeout
            request_id = next(self._request_ids)
            controller.submit(request_id, seat_observation, mask, deadline)
            self._pending_requests[current_seat] = _PendingRequest(
                request_id=request_id,
                deadline=deadline,
                observation=seat_observation,
            )
            return

        while True:
            response = controller.poll()
            if response is None:
                break
            request_id, action = response
            if request_id != pending.request_id:
                continue
            self._queue_action_and_clear(
                current_seat,
                controller,
                action,
                observation=pending.observation,
            )
            return

        if now >= pending.deadline:
            self._queue_action_and_clear(
                current_seat,
                controller,
                None,
                observation=pending.observation,
            )

    def on_stop(self) -> None:
        self._cleanup_game()
        super().on_stop()

    def _initialise_agents(self, human_seats: Sequence[int]) -> None:
        if self.env is None or self.wrapper is None:
            return

        num_players = self.env.num_players
        self._agents = [None] * num_players
        self._fallback_agent = RandomDiscardAgent(env=self.env)

        agent0 = _AIAgent(self.env, backbone="resnet50")
        agent0.load_model("model_weights/latest.pt")

        human_seat_set = set(human_seats)
        for seat in range(num_players):
            if seat in human_seat_set:
                agent = HumanPlayerAgent()
                self.wrapper.bind_human_ui(seat, agent)
                self.wrapper.set_assist_agent(seat, agent0)
            else:
                agent = agent0
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

        self.env = MahjongEnv(num_players=4, num_rounds=8)
        self.wrapper = MahjongEnvKivyWrapper(env=self.env)
        self._pending_requests = {}
        self._request_ids = count()
        self._step_index = 0

        if self._game_screen is not None:
            self._game_screen.clear_widgets()
            self._game_screen.add_widget(self.wrapper.root)
        
        if len(human_seats) > 0:
            self.wrapper.set_focus_player(human_seats[0])

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
        self._start_game(human_seats=(random.choice([0, 1, 2, 3]),))

    def start_ai_vs_ai(self) -> None:
        self._start_game(human_seats=())

    def _queue_action_and_clear(
        self,
        seat: int,
        controller: AgentController,
        action: Optional[Any],
        *,
        observation: Any = None,
    ) -> None:
        pending = self._pending_requests.pop(seat, None)
        controller.flush()

        seat_observation = observation
        if seat_observation is None and pending is not None:
            seat_observation = pending.observation
        if seat_observation is None:
            seat_observation = self._get_seat_observation(seat)

        response = self._normalise_action(seat, action, pending)

        if response is None and self._fallback_agent is not None and seat_observation is not None:
            fallback_action = self._fallback_agent.predict(seat_observation)
            response = self._normalise_action(seat, fallback_action, pending)

        if response is None:
            return
        if self.wrapper is not None:
            self.wrapper.queue_action(response)
            self._step_index += 1

    def _handle_environment_reset(self) -> None:
        self.return_to_menu()

    def _is_done(self, terminations: Any, truncations: Any) -> bool:
        done = False
        if isinstance(terminations, dict):
            done = any(bool(value) for value in terminations.values())
        else:
            done = bool(terminations)
        if isinstance(truncations, dict):
            done = done or any(bool(value) for value in truncations.values())
        else:
            done = done or bool(truncations)
        return done

    def _get_seat_observation(self, seat: int) -> Any:
        if isinstance(self._observation, dict):
            return self._observation.get(seat)
        return self._observation

    def _extract_mask(self, observation: Any) -> Optional[Iterable[Any]]:
        if observation is None:
            return None
        candidate = getattr(observation, "legal_actions_mask", None)
        if candidate is not None:
            return candidate
        if isinstance(observation, dict):
            for key in ("legal_actions_mask", "action_mask", "mask", "legal_mask"):
                if key in observation:
                    return observation[key]
        return None

    def _mask_actionable(self, mask: Any) -> int:
        if mask is None:
            return 0
        if isinstance(mask, dict):
            for key in ("legal_actions_mask", "action_mask", "mask", "legal_mask"):
                if key in mask:
                    mask = mask[key]
                    break
        if isinstance(mask, Iterable):
            return sum(1 for value in mask if bool(value))
        return 0

    def _auto_action_from_mask(self, mask: Any) -> Optional[int]:
        if mask is None:
            return None
        if isinstance(mask, dict):
            for key in ("legal_actions_mask", "action_mask", "mask", "legal_mask"):
                if key in mask:
                    mask = mask[key]
                    break
        if not isinstance(mask, Iterable):
            return None
        indices = [idx for idx, value in enumerate(mask) if bool(value)]
        if len(indices) == 1:
            return indices[0]
        return None

    def _normalise_action(
        self, seat: int, action: Optional[Any], pending: Optional[_PendingRequest]
    ) -> Optional[Response]:
        if action is None:
            return None
        if isinstance(action, Response):
            return action
        sketch: Optional[ActionSketch]
        if isinstance(action, ActionSketch):
            sketch = action
        elif isinstance(action, int):
            action_id = int(action)
            action_type = get_action_type_from_index(action_id)
            if action_type is None:
                return None
            sketch = ActionSketch(
                action_type=action_type,
                payload={"action_id": action_id},
            )
        else:
            return None

        request_id = (
            str(pending.request_id)
            if pending is not None
            else f"auto-{self._step_index}"
        )
        step_id = self._step_index
        try:
            seat_enum = Seat(seat)
        except ValueError:
            seat_enum = Seat(seat % len(Seat))
        return Response(
            room_id=self._room_id,
            step_id=step_id,
            request_id=request_id,
            from_seat=seat_enum,
            chosen=sketch,
        )

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
        self._step_index = 0
        self._pending_requests = {}
        self._request_ids = count()
        if self._game_screen is not None:
            self._game_screen.clear_widgets()
        if self._screen_manager is not None:
            self._screen_manager.current = "start_menu"


if __name__ == "__main__":
    MahjongKivyApp().run()
