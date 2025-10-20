import os
import queue
import sys
import threading
import time
import random
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

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
from kivy.uix.togglebutton import ToggleButton

from agent.agent_card import AgentCard
from agent.human_player_agent import HumanPlayerAgent
from agent.visual_agent import VisualAgent as _AIAgent
#from agent.rule_based_agent import RuleBasedAgent as _AIAgent
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
        self._agent_selection_screen: Optional[Screen] = None
        self._drive_event = None
        self._agent_cards: dict[str, AgentCard] = {}
        self._default_agent_card_id: Optional[str] = None
        self._selected_agent_card_id: Optional[str] = None
        self._pending_human_seats: Sequence[int] = ()
        self._agent_cards = self._load_agent_cards()

    def build(self):
        base_path = Path(__file__).resolve().parent
        layout_path = base_path / "assets" / "layout" / "mahjong_gui.kv"
        if layout_path.exists():
            Builder.load_file(str(layout_path))

        self._screen_manager = ScreenManager()
        start_menu = Factory.StartMenuScreen(name="start_menu")
        self._agent_selection_screen = Factory.AgentSelectionScreen(
            name="agent_selection"
        )
        self._game_screen = Screen(name="game")
        self._screen_manager.add_widget(start_menu)
        if self._agent_selection_screen is not None:
            self._screen_manager.add_widget(self._agent_selection_screen)
        self._screen_manager.add_widget(self._game_screen)

        self._populate_agent_selection()

        return self._screen_manager

    def _load_agent_cards(self) -> dict[str, AgentCard]:
        base_path = Path(__file__).resolve().parent
        cards: dict[str, AgentCard] = {}

        model_path_rel = Path("model_weights") / "latest.pt"
        default_model_path = base_path / model_path_rel
        preview_image = Path("assets") / "texture" / "main_menu.png"
        agent = _AIAgent(backbone="resnet50")

        if default_model_path.exists():
            try:
                agent.load_model(str(default_model_path))
            except Exception as exc:  # pragma: no cover - defensive logging
                print(
                    f"Failed to load agent model '{default_model_path}': {exc}",
                    file=sys.stderr,
                )

        identifier = "resnet50_latest"
        cards[identifier] = AgentCard(
            identifier=identifier,
            title="ResNet50 Latest",
            model_path=str(model_path_rel),
            preview_image=str(preview_image),
            agent=agent,
        )

        if self._default_agent_card_id is None:
            self._default_agent_card_id = identifier

        return cards

    def _resolve_selected_agent_card_id(self) -> Optional[str]:
        if not self._agent_cards:
            return None
        if (
            self._selected_agent_card_id
            and self._selected_agent_card_id in self._agent_cards
        ):
            return self._selected_agent_card_id
        if (
            self._default_agent_card_id
            and self._default_agent_card_id in self._agent_cards
        ):
            return self._default_agent_card_id
        return next(iter(self._agent_cards))

    def _get_selected_agent_card(self) -> Optional[AgentCard]:
        identifier = self._resolve_selected_agent_card_id()
        if identifier is None:
            return None
        return self._agent_cards.get(identifier)

    def select_agent_card(self, identifier: str) -> None:
        if identifier in self._agent_cards:
            self._selected_agent_card_id = identifier

    def _populate_agent_selection(self) -> None:
        if self._agent_selection_screen is None:
            return
        ids = self._agent_selection_screen.ids
        container = ids.get("cards_container")
        if container is None:
            return
        container.clear_widgets()

        selected_id = self._resolve_selected_agent_card_id()

        for identifier, card in self._agent_cards.items():
            button = ToggleButton(
                text=card.title,
                size_hint_y=None,
                height=120,
                group="agent_cards",
                halign="center",
                valign="middle",
            )
            button.padding = (20, 20)

            def _update_text_size(btn: ToggleButton, *_args: Any) -> None:
                btn.text_size = (btn.width - 40, None)

            _update_text_size(button)
            button.bind(size=_update_text_size)
            setattr(button, "card_id", identifier)
            if identifier == selected_id:
                button.state = "down"
            button.bind(
                on_release=lambda instance, card_id=identifier: self.choose_agent_card(
                    card_id
                )
            )
            container.add_widget(button)

        self._update_agent_selection_controls()

    def _update_agent_selection_controls(self) -> None:
        if self._agent_selection_screen is None:
            return
        ids = self._agent_selection_screen.ids
        selection_label = ids.get("selection_label")
        confirm_button = ids.get("confirm_button")
        card = self._get_selected_agent_card()
        if selection_label is not None:
            if not self._agent_cards:
                selection_label.text = "No AI opponents available"
            elif card is not None:
                selection_label.text = f"Selected opponent: {card.title}"
            else:
                selection_label.text = "Select an AI opponent"
        if confirm_button is not None:
            confirm_button.disabled = card is None

    def _update_agent_selection_button_states(self) -> None:
        if self._agent_selection_screen is None:
            return
        ids = self._agent_selection_screen.ids
        container = ids.get("cards_container")
        selected_id = self._resolve_selected_agent_card_id()
        if container is not None:
            for child in container.children:
                if isinstance(child, ToggleButton):
                    card_id = getattr(child, "card_id", None)
                    child.state = "down" if card_id == selected_id else "normal"
        self._update_agent_selection_controls()

    def choose_agent_card(self, identifier: str) -> None:
        if identifier not in self._agent_cards:
            return
        self._selected_agent_card_id = identifier
        self._update_agent_selection_button_states()

    def begin_human_match_with_card(self) -> None:
        card = self._get_selected_agent_card()
        if card is None:
            return
        self._selected_agent_card_id = card.identifier
        if not self._pending_human_seats:
            self._pending_human_seats = (random.choice([0, 1, 2, 3]),)
        human_seats = self._pending_human_seats
        self._pending_human_seats = ()
        self._start_game(human_seats=human_seats)

    def cancel_agent_selection(self) -> None:
        self._pending_human_seats = ()
        if self._screen_manager is not None:
            self._screen_manager.current = "start_menu"

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

        selected_card = self._get_selected_agent_card()
        shared_ai_agent: Optional[Any] = None
        if selected_card is not None:
            shared_ai_agent = selected_card.agent
            if self.env is not None and shared_ai_agent is not None:
                set_env = getattr(shared_ai_agent, "set_environment", None)
                if callable(set_env):
                    set_env(self.env)
                elif hasattr(shared_ai_agent, "env"):
                    setattr(shared_ai_agent, "env", self.env)

        human_seat_set = set(human_seats)
        for seat in range(num_players):
            if seat in human_seat_set:
                agent = HumanPlayerAgent()
                self.wrapper.bind_human_ui(seat, agent)
                if shared_ai_agent is not None:
                    self.wrapper.set_assist_agent(seat, shared_ai_agent)
            else:
                agent = shared_ai_agent if shared_ai_agent is not None else self._fallback_agent
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
        if self._screen_manager is None:
            return
        self._pending_human_seats = (random.choice([0, 1, 2, 3]),)
        self._populate_agent_selection()
        self._screen_manager.current = "agent_selection"

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
        selected_card = self._get_selected_agent_card()
        if selected_card is not None:
            set_env = getattr(selected_card.agent, "set_environment", None)
            if callable(set_env):
                set_env(None)
            elif hasattr(selected_card.agent, "env"):
                setattr(selected_card.agent, "env", None)
        self.env = None
        self._observation = None
        self._pending_requests = {}
        self._request_ids = count()
        self._pending_human_seats = ()
        if self._game_screen is not None:
            self._game_screen.clear_widgets()
        if self._screen_manager is not None:
            self._screen_manager.current = "start_menu"


if __name__ == "__main__":
    MahjongKivyApp().run()
