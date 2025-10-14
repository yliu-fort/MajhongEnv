# Mahjong Environment Refactor Suggestions

## Pain Points in `src/mahjong_env.py`
- **Single, monolithic class** (`MahjongEnvBase`) mixes round setup, per-turn logic, scoring, and helper utilities in one file of ~1.6k lines, making it difficult to reason about or test individual responsibilities.
- **`step` method branching** handles every phase transition inline, so state transitions, validation, and logging are tightly coupled and hard to reuse or unit test in isolation.【F:src/mahjong_env.py†L112-L521】
- **Shared mutable state** (e.g., `claims`, `selected_tiles`, `machii`, `riichi` flags) is mutated in multiple methods without a central authority, increasing the risk of inconsistent updates.【F:src/mahjong_env.py†L205-L676】
- **Utility helpers** (tile conversion, mask generation, shanten/hand checks) live alongside environment control flow, obscuring the main gameplay logic.【F:src/mahjong_env.py†L1040-L1359】

## Suggested Module Boundaries
1. **`state/round_state.py`** – dataclasses describing persistent table state (scores, round/hand counters, dora indicators) and per-hand runtime state (hands, discards, melds). Encapsulate mutations behind intent-revealing methods (e.g., `start_new_round()`, `apply_score_changes()`), so other modules request state transitions rather than mutating raw lists.
2. **`engine/phase_controller.py`** – orchestrates phase progression (`draw`, `discard`, `claim`, `score`, etc.). Break the current `step` branches into smaller strategies (one per phase) that accept the current state and return the next phase plus side effects. This makes it easier to unit test each phase independently.【F:src/mahjong_env.py†L196-L676】
3. **`engine/claim_manager.py`** – centralize claim discovery and resolution (`check_self_claim`, `check_others_claim`, `can_ron`, `can_kan`, `can_chi`, etc.). Provide pure functions that accept immutable snapshots of the table state and produce ordered claim queues. This isolates priority rules and furiten checks from the main loop.【F:src/mahjong_env.py†L704-L1112】
4. **`engine/action_masks.py`** – expose standalone functions that derive legal moves for a phase. These functions can then be called by the environment and by agents/tests without coupling to the full `MahjongEnv` object.【F:src/mahjong_env.py†L1367-L1494】
5. **`engine/scoring.py`** – wrap `agari_calculation`, riichi settlement, and ryūkyoku penalties. This module would convert raw `MahjongHandChecker` output into score deltas and logging payloads, removing score logic from the environment core.【F:src/mahjong_env.py†L480-L594】【F:src/mahjong_env.py†L804-L874】
6. **`engine/draw.py`** – small module handling deck/dead wall mutations (`draw_tile`, `draw_tile_from_dead_wall`) so future rule variations (e.g., alternative wall composition) can be injected or mocked easily.【F:src/mahjong_env.py†L876-L930】
7. **`utils/tiles.py`** – move tile conversion helpers (`tiles_136_to_bool`, `get_claim_tile_mask`, etc.) into a shared utility package. This keeps the environment class focused on gameplay sequencing while providing reusable primitives for agents or analysis scripts.【F:src/mahjong_env.py†L1118-L1359】

## Implementation Approach
- **Introduce data classes** for immutable snapshots (e.g., `PlayerState`, `RoundContext`). Pass these between modules instead of sharing raw lists. This change alone clarifies who owns which state transitions.
- **Extract by feature**: start by moving the tile utility helpers and action-mask functions into new modules since they have minimal dependencies. Then gradually migrate claim/scoring logic, replacing direct attribute access with method calls on the new classes.
- **Add integration tests** for each phase controller (draw→discard→claim) before moving logic, ensuring behavior stays consistent across refactors.
- **Leverage dependency injection**: allow the environment to accept collaborators (claim manager, scoring service, logger) so simulation variants can swap implementations without editing the monolithic class.

## Benefits
- Easier to understand and test because each module owns a narrow concern.
- Clearer extension points for alternative rulesets (e.g., optional kuikae, sudden death) without touching core environment flow.
- Reduced risk of regression when tweaking scoring or claim rules; changes stay local to the relevant module.
- Improved agent development: standalone action-mask and state snapshot utilities can be reused for offline training or rule validation.
