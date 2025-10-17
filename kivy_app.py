import os
import queue
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "agent"))

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.uix.screenmanager import Screen, ScreenManager

from agent.human_player_agent import HumanPlayerAgent
from agent.visual_agent import VisualAgent
from agent.random_discard_agent import RandomDiscardAgent
from mahjong_env import MahjongEnv
from mahjong_wrapper_kivy import MahjongEnvKivyWrapper
from mahjong_features import get_action_index
from tenhou_to_mahjong import TenhouMeld, tid136_to_t34


@dataclass
class _PendingRequest:
    request_id: int
    deadline: float


@dataclass
class ReplayEvent:
    seat: Optional[int]
    kind: str
    data: Dict[str, Any]


class ReplayScript:
    """Sequence of actions reconstructed from a Tenhou mjlog."""

    def __init__(self, events: Sequence[ReplayEvent], seed: Optional[str] = None):
        self._events: Deque[ReplayEvent] = deque(events)
        self.seed = seed

    @staticmethod
    def from_log_text(log_text: str) -> "ReplayScript":
        root = ET.fromstring(log_text)
        events: list[ReplayEvent] = []
        seed: Optional[str] = None
        last_discard: Optional[ReplayEvent] = None

        for elem in root:
            tag = elem.tag
            if tag == "SHUFFLE":
                raw_seed = elem.attrib.get("seed")
                if raw_seed:
                    seed = raw_seed.split(",")[-1]
            elif tag and tag[0] in "TUVW" and tag[1:].isdigit():
                who = "TUVW".index(tag[0])
                tile = int(tag[1:])
                events.append(ReplayEvent(seat=who, kind="draw", data={"tile": tile}))
            elif tag and tag[0] in "DEFG" and tag[1:].isdigit():
                who = "DEFG".index(tag[0])
                tile = int(tag[1:])
                last_discard = ReplayEvent(
                    seat=who,
                    kind="discard",
                    data={"tile": tile, "riichi": False},
                )
                events.append(last_discard)
            elif tag == "REACH":
                who = int(elem.attrib.get("who", "0"))
                step = int(elem.attrib.get("step", "0"))
                if step == 1 and last_discard is not None and last_discard.seat == who:
                    last_discard.data["riichi"] = True
                events.append(
                    ReplayEvent(seat=who, kind="reach", data={"step": step})
                )
            elif tag == "N":
                who = int(elem.attrib.get("who", "0"))
                meld_val = int(elem.attrib.get("m", "0"))
                meld = TenhouMeld(who, meld_val)
                kind = meld.type or "meld"
                events.append(
                    ReplayEvent(seat=who, kind=kind, data={"meld": meld})
                )
            elif tag == "DORA":
                tile = int(elem.attrib.get("hai", "0"))
                events.append(ReplayEvent(seat=None, kind="dora", data={"tile": tile}))
            elif tag == "AGARI":
                who = int(elem.attrib.get("who", "0"))
                from_who = int(elem.attrib.get("fromWho", str(who)))
                kind = "tsumo" if who == from_who else "ron"
                events.append(ReplayEvent(seat=who, kind=kind, data=dict(elem.attrib)))
            elif tag == "RYUUKYOKU":
                events.append(ReplayEvent(seat=None, kind="ryuukyoku", data=dict(elem.attrib)))

        return ReplayScript(events, seed=seed)

    def next_action(self, seat: int, phase: str) -> Optional[int]:
        while self._events:
            event = self._events[0]
            if event.kind in {"draw", "dora", "reach"}:
                self._events.popleft()
                continue
            if event.kind == "ryuukyoku" and phase != "ryuukyoku":
                self._events.popleft()
                continue
            if event.seat is not None and event.seat != seat:
                return self._pass_action_for_phase(phase)
            event = self._events.popleft()
            return self._map_event_to_action(event, phase)
        return self._pass_action_for_phase(phase)

    def _map_event_to_action(self, event: ReplayEvent, phase: str) -> Optional[int]:
        if event.kind == "discard":
            tile_t34 = tid136_to_t34(event.data["tile"])
            if event.data.get("riichi"):
                return get_action_index(tile_t34, "riichi" if phase == "riichi" else "discard")
            return get_action_index(tile_t34, "discard")
        if event.kind in {"chi", "pon", "kan", "chakan"}:
            meld: TenhouMeld = event.data["meld"]
            if event.kind == "chi":
                payload = (meld.base_t34 or 0, meld.called_index or 0)
            elif event.kind == "pon":
                payload = (meld.base_t34 or 0, meld.called_index or 0)
            elif event.kind == "kan":
                called = None if not meld.opened else meld.called_index or 0
                payload = (meld.base_t34 or 0, called)
            else:  # chakan
                payload = (meld.base_t34 or 0, meld.called_index or 0)
            return get_action_index(payload, event.kind)
        if event.kind in {"ron", "tsumo"}:
            return get_action_index(None, event.kind)
        if event.kind == "ryuukyoku":
            return get_action_index(None, "ryuukyoku")
        return None

    @staticmethod
    def _pass_action_for_phase(phase: str) -> Optional[int]:
        passable = {
            "chi",
            "pon",
            "kan",
            "chakan",
            "ankan",
            "riichi",
            "ron",
            "tsumo",
            "ryuukyoku",
        }
        if phase in passable:
            return get_action_index(None, ("pass", phase))
        return None


class ReplayAgent:
    def __init__(self, seat: int, script: ReplayScript):
        self._seat = seat
        self._script = script

    def predict(self, observation: Any) -> Optional[int]:
        phase = ""
        if isinstance(observation, tuple) and len(observation) > 1:
            metadata = observation[1]
            if isinstance(metadata, dict):
                phase = metadata.get("phase", "")
        action = self._script.next_action(self._seat, phase)
        return action


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


class MahjongKivyApp(App):
    """Entry point for the Kivy Mahjong visualiser."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._controllers: list[AgentController] = []
        self._agents: list[Any] = []
        self._fallback_agent: RandomDiscardAgent = None
        self._pending_requests: dict[int, _PendingRequest] = {}
        self._request_ids = count()
        self._action_timeout = 5.0
        self._observation: Any = None
        self.env: Optional[MahjongEnv] = None
        self.wrapper: Optional[MahjongEnvKivyWrapper] = None
        self._drive_event = None
        self.screen_manager: Optional[ScreenManager] = None
        self._start_menu_screen: Optional[Screen] = None
        self._game_screen: Optional[Screen] = None

    def build(self):
        base_path = Path(__file__).resolve().parent
        layout_path = base_path / "assets" / "layout" / "mahjong_gui.kv"
        if layout_path.exists():
            Builder.load_file(str(layout_path))
        self.screen_manager = ScreenManager()
        self._start_menu_screen = Factory.StartMenuScreen()
        self._start_menu_screen.name = "start_menu"
        self._game_screen = Screen(name="game")
        self.screen_manager.add_widget(self._start_menu_screen)
        self.screen_manager.add_widget(self._game_screen)
        self.screen_manager.current = "start_menu"
        return self.screen_manager

    def _drive_environment(self, _dt: float) -> None:
        if not self.wrapper or not self.env:
            return

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
        if not self._controllers:
            return
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
        self._shutdown_controllers()
        if self.wrapper is not None:
            self.wrapper.close()
        super().on_stop()

    def start_ai_vs_human(self) -> None:
        self._start_game("ai_vs_human")

    def start_ai_vs_ai(self) -> None:
        self._start_game("ai_vs_ai")

    def start_replay(self) -> None:
        script = self._load_replay_script()
        if script is None:
            return
        self._start_game("replay", script=script)

    def _start_game(
        self,
        mode: str,
        *,
        script: Optional[ReplayScript] = None,
    ) -> None:
        self._shutdown_controllers()
        if self.wrapper is not None:
            self.wrapper.close()

        self.env = MahjongEnv(num_players=4)
        if script is not None and script.seed:
            self.env.set_seed(script.seed)
        else:
            self.env.set_seed(None)
        self.wrapper = MahjongEnvKivyWrapper(env=self.env)
        self._configure_agents(mode, script)
        self._initialise_controllers()
        self._pending_requests.clear()
        self._request_ids = count()
        self._observation = self.wrapper.reset()
        self._start_controllers()
        interval = 1.0 / max(1, self.wrapper.fps)
        if self._drive_event is not None:
            self._drive_event.cancel()
        self._drive_event = Clock.schedule_interval(self._drive_environment, interval)
        self._attach_wrapper_to_game_screen()
        if self.screen_manager is not None:
            self.screen_manager.current = "game"

    def _configure_agents(
        self, mode: str, script: Optional[ReplayScript]
    ) -> None:
        assert self.env is not None
        assert self.wrapper is not None

        num_players = self.env.num_players
        self._agents = [None] * num_players
        if mode == "ai_vs_human":
            human_agent = HumanPlayerAgent()
            self._agents[0] = human_agent
            self.wrapper.bind_human_ui(0, human_agent)
            for seat in range(1, num_players):
                self._agents[seat] = self._create_visual_agent()
            self._fallback_agent = RandomDiscardAgent(env=self.env)
        elif mode == "ai_vs_ai":
            for seat in range(num_players):
                self._agents[seat] = self._create_visual_agent()
            self._fallback_agent = RandomDiscardAgent(env=self.env)
        elif mode == "replay":
            if script is None:
                raise ValueError("Replay mode requires a script")
            for seat in range(num_players):
                self._agents[seat] = ReplayAgent(seat, script)
            self._fallback_agent = RandomDiscardAgent(env=self.env)
        else:
            raise ValueError(f"Unknown game mode: {mode}")

    def _create_visual_agent(self) -> VisualAgent:
        agent = VisualAgent(self.env, backbone="resnet50")
        agent.load_model("model_weights/latest.pt")
        return agent

    def _attach_wrapper_to_game_screen(self) -> None:
        if self._game_screen is None or self.wrapper is None:
            return
        self._game_screen.clear_widgets()
        self._game_screen.add_widget(self.wrapper.root)

    def _load_replay_script(self) -> Optional[ReplayScript]:
        base_path = Path(__file__).resolve().parent
        replay_dir = base_path / "log_analyser" / "paipu"
        candidates = []
        if replay_dir.exists():
            candidates = sorted(
                replay_dir.glob("*.mjlog"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        if not candidates:
            return None
        try:
            log_text = candidates[0].read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return ReplayScript.from_log_text(log_text)
        except ET.ParseError:
            return None

    def _initialise_controllers(self) -> None:
        self._controllers = [
            AgentController(seat=index, agent=self._agents[index])
            for index in range(self.env.num_players)
        ]

    def _start_controllers(self) -> None:
        for controller in self._controllers:
            controller.start()

    def _queue_action_and_clear(
        self, seat: int, controller: AgentController, action: Optional[int]
    ) -> None:
        controller.flush()
        self._pending_requests.pop(seat, None)
        if action is None and self._fallback_agent is not None:
            action = self._fallback_agent.predict(self._observation)
        if action is None:
            return
        self.wrapper.queue_action(action)

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
        self._controllers = []
        if self._drive_event is not None:
            self._drive_event.cancel()
            self._drive_event = None


if __name__ == "__main__":
    MahjongKivyApp().run()
