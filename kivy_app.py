import os
import queue
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "agent"))

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.uix.screenmanager import ScreenManager

from agent.human_player_agent import HumanPlayerAgent
from agent.visual_agent import VisualAgent
from agent.random_discard_agent import RandomDiscardAgent
from mahjong_env import MahjongEnv
from mahjong_wrapper_kivy import MahjongEnvKivyWrapper, _RenderPayload
from tenhou_to_mahjong import TenhouMeld


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


class MahjongKivyApp(App):
    """Entry point for the Kivy Mahjong visualiser."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.env: Optional[MahjongEnv] = None
        self.wrapper: Optional[MahjongEnvKivyWrapper] = None
        self._controllers: list[AgentController] = []
        self._agents: list[Any] = []
        self._fallback_agent: RandomDiscardAgent = None
        self._pending_requests: dict[int, _PendingRequest] = {}
        self._request_ids = count()
        self._action_timeout = 5.0
        self._observation: Any = None
        self._drive_event = None
        self._replay_event = None
        self._replay_frames: list[dict[str, Any]] = []
        self._replay_index = 0
        self._screen_manager: Optional[ScreenManager] = None
        self._game_screen = None

    def build(self):
        base_path = Path(__file__).resolve().parent
        layout_path = base_path / "assets" / "layout" / "mahjong_gui.kv"
        if layout_path.exists():
            Builder.load_file(str(layout_path))
        manager = ScreenManager()
        start_menu = Factory.StartMenuScreen()
        game_screen = Factory.GameScreen()
        manager.add_widget(start_menu)
        manager.add_widget(game_screen)
        manager.current = getattr(start_menu, "name", "start_menu")
        self._screen_manager = manager
        self._game_screen = game_screen
        return manager

    def _cancel_drive_event(self) -> None:
        if self._drive_event is not None:
            try:
                self._drive_event.cancel()
            finally:
                self._drive_event = None

    def _cancel_replay_event(self) -> None:
        if self._replay_event is not None:
            try:
                self._replay_event.cancel()
            finally:
                self._replay_event = None

    def _cleanup_environment(self) -> None:
        self._cancel_drive_event()
        self._cancel_replay_event()
        if self._controllers:
            self._shutdown_controllers()
        self._controllers = []
        self._pending_requests.clear()
        if self.wrapper is not None:
            try:
                self.wrapper.close()
            except Exception:
                pass
        self.wrapper = None
        self.env = None
        self._replay_frames = []
        self._replay_index = 0
        self._agents = []

    def _start_game(self, mode: str) -> None:
        self._cleanup_environment()
        self.env = MahjongEnv(num_players=4)
        self.wrapper = MahjongEnvKivyWrapper(env=self.env)
        if self._game_screen is not None:
            self._game_screen.clear_widgets()
            self._game_screen.add_widget(self.wrapper.root)
        self._pending_requests.clear()
        self._request_ids = count()
        self._observation = None
        self._fallback_agent = RandomDiscardAgent(env=self.env)
        self._agents = [None] * self.env.num_players
        if mode != "replay":
            self._initialise_agents()
            if mode == "ai_vs_human":
                self._initialise_human()
            self._initialise_controllers()
            self._observation = self.wrapper.reset()
            self._start_controllers()
            interval = 1.0 / max(1, getattr(self.wrapper, "fps", 1))
            self._drive_event = Clock.schedule_interval(self._drive_environment, interval)
        else:
            self.wrapper.reset()
            self.env.done = False
            self._controllers = []

    def start_ai_vs_human(self) -> None:
        self._start_game("ai_vs_human")
        if self._screen_manager is not None:
            self._screen_manager.current = "game"

    def start_ai_vs_ai(self) -> None:
        self._start_game("ai_vs_ai")
        if self._screen_manager is not None:
            self._screen_manager.current = "game"

    def start_replay(self, log_path: Optional[str] = None) -> None:
        target_path = Path(log_path) if log_path else self._find_latest_log()
        if target_path is None or not target_path.exists():
            print("No replay log found.")
            return
        try:
            log_text = target_path.read_text(encoding="utf-8")
        except Exception:
            print(f"Failed to read replay log from {target_path}")
            return
        frames = self._load_replay_frames(log_text)
        if not frames:
            print(f"Replay log {target_path} did not contain any frames.")
            return
        self._start_game("replay")
        self._replay_frames = frames
        self._replay_index = 0
        self._apply_replay_frame(frames[0])
        self._start_replay_loop()
        if self._screen_manager is not None:
            self._screen_manager.current = "game"

    def _start_replay_loop(self) -> None:
        self._cancel_replay_event()
        if not self._replay_frames:
            return
        interval = 0.75
        self._replay_event = Clock.schedule_interval(self._advance_replay, interval)

    def _advance_replay(self, _dt: float) -> bool:
        if not self._replay_frames:
            return False
        if self._replay_index + 1 >= len(self._replay_frames):
            self._cancel_replay_event()
            return False
        self._replay_index += 1
        self._apply_replay_frame(self._replay_frames[self._replay_index])
        return True

    def _find_latest_log(self) -> Optional[Path]:
        candidate_dirs = [
            Path(__file__).resolve().parent / "log",
            Path(__file__).resolve().parent.parent / "log_analyser" / "paipu",
        ]
        log_files: list[Path] = []
        for directory in candidate_dirs:
            if directory.exists():
                log_files.extend(sorted(directory.glob("*.mjlog")))
        if not log_files:
            return None
        return max(log_files, key=lambda path: path.stat().st_mtime)

    def _load_replay_frames(self, log_text: str) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        try:
            root = ET.fromstring(log_text)
        except ET.ParseError:
            return frames

        num_players = 4
        hands: list[list[int]] = [[] for _ in range(num_players)]
        rivers: list[list[int]] = [[] for _ in range(num_players)]
        melds: list[list[dict[str, Any]]] = [[] for _ in range(num_players)]
        scores = [250 for _ in range(num_players)]
        riichi = [False for _ in range(num_players)]
        round_info = [0, 0]
        num_riichi = 0
        kyoutaku = 0
        dora_indicator: list[int] = []
        current_player = 0
        phase = "draw"
        done = False
        message = ""

        def clone_melds() -> list[list[dict[str, Any]]]:
            result: list[list[dict[str, Any]]] = []
            for seat_melds in melds:
                seat_copy: list[dict[str, Any]] = []
                for entry in seat_melds:
                    seat_copy.append(
                        {
                            "type": entry.get("type"),
                            "fromwho": entry.get("fromwho"),
                            "offset": entry.get("offset"),
                            "m": list(entry.get("m", [])),
                            "claimed_tile": entry.get("claimed_tile"),
                            "opened": entry.get("opened", True),
                        }
                    )
                result.append(seat_copy)
            return result

        def snapshot() -> None:
            frames.append(
                {
                    "hands": [list(hand) for hand in hands],
                    "discard_pile_seq": [list(river) for river in rivers],
                    "melds": clone_melds(),
                    "current_player": current_player,
                    "phase": phase,
                    "scores": list(scores),
                    "riichi": list(riichi),
                    "round": list(round_info),
                    "num_riichi": num_riichi,
                    "num_kyoutaku": kyoutaku,
                    "dora_indicator": list(dora_indicator),
                    "done": done,
                    "message": message,
                }
            )

        for element in root:
            tag = element.tag
            if tag == "INIT":
                done = False
                message = ""
                seed = [int(x) for x in element.get("seed", "0,0,0,0,0,0").split(",")]
                round_info[0] = seed[0]
                round_info[1] = seed[1]
                kyoutaku = seed[2]
                dora_indicator = [seed[5]] if len(seed) > 5 else []
                scores = [int(x) for x in element.get("ten", "250,250,250,250").split(",")]
                for seat in range(num_players):
                    key = f"hai{seat}"
                    value = element.get(key)
                    if value:
                        hands[seat] = [int(x) for x in value.split(",") if x]
                    else:
                        hands[seat] = []
                    rivers[seat] = []
                    melds[seat] = []
                    riichi[seat] = False
                current_player = int(element.get("oya", "0"))
                num_riichi = 0
                snapshot()
                continue

            if tag and tag[0] in "TUVW":
                actor = "TUVW".index(tag[0])
                tile = int(tag[1:])
                hands[actor].append(tile)
                current_player = actor
                phase = "draw"
                snapshot()
                continue

            if tag and tag[0] in "DEFG":
                actor = "DEFG".index(tag[0])
                tile = int(tag[1:])
                removed = False
                if tile in hands[actor]:
                    hands[actor].remove(tile)
                    removed = True
                if not removed:
                    tile_34 = tile // 4
                    for idx, value in enumerate(hands[actor]):
                        if value // 4 == tile_34:
                            hands[actor].pop(idx)
                            removed = True
                            break
                rivers[actor].append(tile)
                current_player = actor
                phase = "discard"
                snapshot()
                continue

            if tag == "N":
                who = int(element.get("who", "0"))
                meld_value = int(element.get("m", "0"))
                meld = TenhouMeld(who, meld_value)
                fromwho = meld.from_who if meld.from_who is not None else who
                if meld.type == "chakan":
                    for existing in melds[who]:
                        claimed_tile = existing.get("claimed_tile")
                        if (
                            existing.get("type") == "pon"
                            and claimed_tile is not None
                            and claimed_tile // 4 == (meld.base_t34 or -1)
                        ):
                            existing["type"] = "chakan"
                            existing["m"].append(meld.tiles_t136[-1])
                            break
                else:
                    tiles = list(meld.tiles_t136)
                    for tile in tiles:
                        if tile in hands[who]:
                            hands[who].remove(tile)
                        else:
                            tile_34 = tile // 4
                            for idx, value in enumerate(hands[who]):
                                if value // 4 == tile_34:
                                    hands[who].pop(idx)
                                    break
                    if fromwho != who and rivers[fromwho]:
                        rivers[fromwho].pop()
                    new_meld = {
                        "type": meld.type,
                        "fromwho": fromwho,
                        "offset": (fromwho - who) % num_players,
                        "m": tiles,
                        "claimed_tile": meld.base_t136 if fromwho != who else None,
                        "opened": fromwho != who or meld.type != "kan",
                    }
                    melds[who].append(new_meld)
                current_player = who
                phase = "discard"
                snapshot()
                continue

            if tag == "REACH":
                who = int(element.get("who", "0"))
                step = element.get("step")
                if step == "2":
                    riichi[who] = True
                    num_riichi += 1
                snapshot()
                continue

            if tag == "DORA":
                tile = element.get("hai")
                if tile is not None:
                    dora_indicator.append(int(tile))
                snapshot()
                continue

            if tag == "RYUUKYOKU":
                sc = element.get("sc")
                if sc:
                    values = [int(x) for x in sc.split(",") if x]
                    scores = [values[i] for i in range(0, len(values), 2)]
                done = True
                message = "Ryuukyoku"
                num_riichi = 0
                snapshot()
                continue

            if tag == "AGARI":
                sc = element.get("sc")
                if sc:
                    values = [int(x) for x in sc.split(",") if x]
                    scores = [values[i] for i in range(0, len(values), 2)]
                owari = element.get("owari")
                if owari:
                    values = [int(x) for x in owari.split(",") if x]
                    if values:
                        scores = [values[i] for i in range(0, len(values), 2)]
                done = True
                message = "Agari"
                num_riichi = 0
                snapshot()
                continue

        return frames

    def _apply_replay_frame(self, frame: dict[str, Any]) -> None:
        if self.env is None or self.wrapper is None:
            return
        hands = frame.get("hands", [])
        discard_seq = frame.get("discard_pile_seq", [])
        melds = frame.get("melds", [])
        num_players = self.env.num_players

        def ensure_length(values: list, default_factory) -> list:
            if len(values) < num_players:
                values.extend(default_factory() for _ in range(num_players - len(values)))
            return values[:num_players]

        self.env.hands = ensure_length([list(hand) for hand in hands], lambda: [])
        self.env.discard_pile_seq = ensure_length([list(seq) for seq in discard_seq], lambda: [])
        self.env.melds = ensure_length(
            [
                [
                    {
                        "type": entry.get("type"),
                        "fromwho": entry.get("fromwho"),
                        "offset": entry.get("offset"),
                        "m": list(entry.get("m", [])),
                        "claimed_tile": entry.get("claimed_tile"),
                        "opened": entry.get("opened", True),
                    }
                    for entry in seat
                ]
                for seat in melds
            ],
            lambda: [],
        )
        self.env.current_player = frame.get("current_player", 0)
        self.env.phase = frame.get("phase", "draw")
        self.env.scores = list(frame.get("scores", getattr(self.env, "scores", [250] * num_players)))
        self.env.riichi = list(frame.get("riichi", getattr(self.env, "riichi", [False] * num_players)))
        self.env.round = list(frame.get("round", getattr(self.env, "round", [0, 0])))
        self.env.num_riichi = frame.get("num_riichi", getattr(self.env, "num_riichi", 0))
        self.env.num_kyoutaku = frame.get("num_kyoutaku", getattr(self.env, "num_kyoutaku", 0))
        self.env.dora_indicator = list(frame.get("dora_indicator", getattr(self.env, "dora_indicator", [])))
        self.env.msg = frame.get("message", "")
        self.env.done = frame.get("done", False)
        self._pending_requests.clear()
        self.wrapper._last_payload = _RenderPayload(
            action=None,
            reward=0.0,
            done=self.env.done,
            info={"message": self.env.msg},
        )
        try:
            self.wrapper._render()
        except Exception:
            pass

    def _drive_environment(self, _dt: float) -> None:
        if self.wrapper is None or self.env is None:
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
        if not self._controllers or current_seat >= len(self._controllers):
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
        self._cleanup_environment()
        super().on_stop()

    def _initialise_agents(self) -> None:
        num_players = self.env.num_players
        self._agents = [None] * num_players

        for seat in range(num_players):
            agent = VisualAgent(self.env, backbone="resnet50")
            agent.load_model("model_weights/latest.pt")
            self._agents[seat] = agent
        self._fallback_agent = RandomDiscardAgent(env=self.env)

    def _initialise_human(self) -> None:
        human_agent = HumanPlayerAgent()
        if self.wrapper is not None:
            self.wrapper.bind_human_ui(0, human_agent)
        self._agents[0] = human_agent

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


if __name__ == "__main__":
    MahjongKivyApp().run()
