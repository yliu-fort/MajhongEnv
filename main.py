import os
import queue
import sys
import threading
import time
import random
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, List

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
from mahjong_wrapper_kivy import MahjongEnvKivyWrapper
from mahjong_features import get_action_type_from_index

from my_types import Seat, Request, Response, ActionSketch
from transport import InProcessTransport
from client import Client, CNNStrategy
from engine import RoomEngine


@dataclass
class _PendingRequest:
    request_id: int
    deadline: float


class AgentController:
    """Background worker that evaluates agent policies on demand."""

    def __init__(self, seat: int, agent: Optional[Any]) -> None:
        self.seat = seat
        self.agent = agent
        self._request_queue: "queue.Queue[Optional[Request]]" = queue.Queue() # TODO: 改成 Request
        self._response_queue: "queue.Queue[Response]" = queue.Queue() # TODO: 改成 Response
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
        request: Request
    ) -> None:
        if self._stop_event.is_set():
            return
        self._request_queue.put(request)

    def poll(self) -> Optional[Response]:
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
                req = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if req is None:
                continue

            action: Optional[ActionSketch] = None
            if len(req.actions) == 1:
                action_id = req.actions[0]
                action = ActionSketch(action_type=get_action_type_from_index(action_id), payload={"action_id": action_id})
            elif len(req.actions) > 1:
                if isinstance(self.agent, HumanPlayerAgent):
                    timeout = max(0.0, req.deadline_ms / 1000 - time.monotonic())
                    try:
                        self.agent.begin_turn(req.observation, timeout) # Remove masks as it is inside obs
                        action = self.agent.wait_for_action()
                    except TimeoutError:
                        action = None
                    except Exception:
                        action = None
                elif self.agent is not None:
                    try:
                        action = self.agent.predict(req.observation)
                    except Exception:
                        action = None
            else:
                action = None

            self._response_queue.put(action if action is None else Response(
                room_id=req.room_id,
                step_id=req.step_id,
                request_id=req.request_id,
                from_seat=Seat(self.seat),
                chosen=action
                ))

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
        self._pending_responses: dict[int, Response] = {}
        self._step_ids = count()
        self._action_timeout = 300.0
        self._observation: Any = None
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
            self._observation, _, done, _ = result
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

        ##################################################################################
        ## TODO: The logic here may be updated by the modern Client, Engine, and Transport class
        #current_seat = getattr(self.env, "current_player", 0) # should remove
        #controller = self._controllers[current_seat] # TODO: 现在我们需要循环所有的controller
        pending = self._pending_requests or self._pending_responses # should remove
        now = time.monotonic()

        if not pending: # no need to check for 'pending' now. we should send Requests to everyone.
            deadline = now + self._action_timeout
            step_id = next(self._step_ids)
            #print(step_id)
            for _, controller in enumerate(self._controllers):
                obs = self._observation[_]
                actions = [i for i, flag in enumerate(obs.legal_actions_mask) if flag]
                #print(f"actions[{_}] {actions} pending {self.wrapper.pending_action}")
                if len(actions) > 1:
                    req = Request(
                    room_id="",
                    step_id=step_id,
                    request_id=f"req-{step_id}-{Seat(_)}",
                    to_seat=Seat(_),
                    actions=actions,
                    observation=obs,
                    deadline_ms=deadline * 1000
                    )
                    controller.submit(req)
                    #print(f"Send {req}")
                    self._pending_requests[_] = _PendingRequest(
                    request_id=req.request_id,
                    deadline=deadline,
                    )
                else:
                    self._pending_responses[_] = None
                    self._pending_requests[_] = _PendingRequest(
                    request_id=f"req-{step_id}-{Seat(_)}",
                    deadline=now,
                    )
            return

        #while True: # we should collect Response from everyone.
        for _, req in self._pending_requests.items():
            if _ not in self._pending_responses.keys():
                # Get response nowait
                resp = self._controllers[_].poll()
                if resp != None and resp.request_id == req.request_id:
                    self._pending_responses[_] = resp
                    #print(f"Recv {resp}")
            
        for _, req in self._pending_requests.items():
            if _ not in self._pending_responses.keys():
                if now >= req.deadline:
                    self._pending_response[_] = None

        if len(self._pending_responses.keys()) > 0 and \
           len(self._pending_requests.keys()) == len(self._pending_responses.keys()):
            self._queue_action_and_clear() # send to self.wrapper.pending_action
        
        ##################################################################################

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
        self._step_ids = count()

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

    def _queue_action_and_clear(self) -> None:
        for controller in self._controllers:
            controller.flush()
        self._pending_requests = {}
        for _ in self._pending_responses.keys():
            if self._pending_responses[_] is None and self._fallback_agent is not None:
                self._pending_responses[_] = \
                Response(room_id="", \
                step_id="", \
                request_id="", \
                from_seat=Seat(_), \
                chosen=self._fallback_agent.predict(self._observation[_]))
        if self._pending_responses == {}:
            return
        if self.wrapper is not None:
            self.wrapper.queue_action({k: v for k ,v in self._pending_responses.items()})
        self._pending_responses = {}

    def _handle_environment_reset(self) -> None:
        self.return_to_menu()

    def _flush_pending_requests(self) -> None:
        self._pending_requests.clear()
        for controller in self._controllers:
            controller.flush()
        self._pending_responses.clear()

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
        self._step_ids = count()
        if self._game_screen is not None:
            self._game_screen.clear_widgets()
        if self._screen_manager is not None:
            self._screen_manager.current = "start_menu"


if __name__ == "__main__":
    MahjongKivyApp().run()
