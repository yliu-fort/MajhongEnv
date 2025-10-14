# Mahjong Kivy Wrapper Refactor Notes

## Current pain points in `mahjong_wrapper_kivy.py`
- **UI strings and localization logic sit next to rendering code.** The module mixes large language dictionaries and ordinal helpers with rendering classes, which inflates the file and complicates testing of UI text formatting separately from graphics concerns.【F:src/mahjong_wrapper_kivy.py†L36-L155】
- **Widget wiring, environment orchestration, asset loading, and frame scheduling all live in the `MahjongEnvKivyWrapper` constructor.** This makes the class hard to instantiate in isolation and increases the surface area that every test needs to cover.【F:src/mahjong_wrapper_kivy.py†L369-L466】
- **Layout utilities and rendering helpers (e.g., `_Rect`, `_get_display_order`, `_get_board_rotation_angle`) are buried inside the wrapper.** They are good candidates for pure functions or small classes that could be unit tested without spinning up Kivy, but their current placement keeps them tightly coupled to the monolithic class.【F:src/mahjong_wrapper_kivy.py†L303-L520】

## Suggested module boundaries
- **`localization.py`:** Move `_ORDINAL_FUNCTIONS` and `_LANGUAGE_STRINGS` (along with formatting helpers) into a dedicated module. Provide a small API such as `get_language_strings(code)` and `format_round(language, data)` so UI code requests translated strings instead of handling format dictionaries directly.【F:src/mahjong_wrapper_kivy.py†L89-L155】
- **`assets.py`:** Extract `_load_tile_assets` and related caches into an asset manager that exposes `get_texture(tile_id, size)`. This module can encapsulate SVG/PNG conversions and make it easier to mock out filesystem and CairoSVG dependencies during tests.【F:src/mahjong_wrapper_kivy.py†L429-L463】
- **`layout.py`:** Promote `_Rect` and rotation/display-order helpers to a reusable layout utility. Having pure functions for seat ordering and orientation keeps the GUI class focused on orchestrating widgets.【F:src/mahjong_wrapper_kivy.py†L303-L520】
- **`state.py` or a view-model layer:** Separate turn/score/auto-step state flags (`_auto_advance`, `_pause_on_score`, `_riichi_states`, etc.) into a dataclass that encapsulates transitions. The Kivy wrapper can observe the state object, while logic around score pauses and pending steps stays testable without rendering.【F:src/mahjong_wrapper_kivy.py†L433-L520】
- **`controls.py`:** Group button/spinner wiring and callbacks (`_toggle_auto`, `_toggle_pause`, `_trigger_step_once`, `_on_language_spinner_text`) to clarify event flows and enable injection of fake widgets when writing tests.【F:src/mahjong_wrapper_kivy.py†L503-L563】

## Testing strategy after extraction
- **Unit tests for localization:** Verify ordinal formatting and string templates per language using pure functions from the proposed `localization` module. These tests run without Kivy and catch template regressions quickly.【F:src/mahjong_wrapper_kivy.py†L89-L155】
- **Pure logic tests for layout/state helpers:** With `_get_display_order`, `_get_board_rotation_angle`, and state-transition functions living in standalone modules, write pytest cases that feed in contrived player counts and flag combinations to assert expected outputs.【F:src/mahjong_wrapper_kivy.py†L503-L563】
- **Asset loader tests with fakes:** Use dependency injection so the asset manager accepts a filesystem adapter; inject temporary directories of SVGs/PNGs to confirm scaling, fallback behavior, and cache invalidation work as expected.【F:src/mahjong_wrapper_kivy.py†L429-L463】
- **Wrapper integration tests:** Keep a thin integration layer that can be exercised with Kivy’s test harness or via mocking `Clock.schedule_interval`. Mock `_BaseMahjongEnv` to produce deterministic payloads and assert that callbacks update UI state appropriately.
- **Visual regression or screenshot tests:** After isolating rendering into smaller components, drive them with deterministic data to capture reference images that guard against layout regressions.

## Isolation tactics
1. **Introduce dependency interfaces.** Replace direct imports or hard-coded singletons (e.g., `Clock.schedule_interval`, direct CairoSVG calls) with injectable callables so tests can substitute no-op or synchronous variants.【F:src/mahjong_wrapper_kivy.py†L429-L466】
2. **Adopt a view-model pattern.** Let a plain-Python object translate environment observations into renderable data (player order, tile states, score text). The Kivy layer subscribes to that object, so business rules can be exercised without GUI dependencies.【F:src/mahjong_wrapper_kivy.py†L483-L563】
3. **Encapsulate language selection.** Move spinner updates and font decisions into a language service that exposes `set_language(code)`; emit events or callbacks when the font or strings change, so widgets only handle presentation.
4. **Leverage dataclasses for mutable state.** Group related flags (auto-advance, pauses, pending actions) into coherent structures with explicit methods like `request_step_once()` or `handle_score_panel_visible()` to avoid scattered boolean logic.【F:src/mahjong_wrapper_kivy.py†L433-L563】

## Incremental refactor roadmap
1. Extract localization constants and helpers first—no external dependencies and high readability impact.
2. Introduce an asset manager class; update the wrapper to depend on it through constructor injection so existing behavior stays intact.
3. Pull layout utilities into a separate module and add unit tests for player ordering and rotation math.
4. Define a state/view-model layer that the wrapper uses for environment updates, then migrate button handlers to operate on that layer.
5. Finally, slim down `MahjongEnvKivyWrapper` so it primarily wires together the extracted components and delegates to them, making future features and tests easier to maintain.
