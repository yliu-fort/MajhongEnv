# Mahjong GUI Layout Reference

This document captures the GUI structure, visual design, and interaction flow implemented in `src/mahjong_wrapper.py`. Use it as a blueprint to recreate the pygame front-end without inspecting the source directly.

## 1. Window Lifecycle & Core Surfaces
- The wrapper subclasses the base environment and instantiates a resizable pygame window during `__init__` before delegating to the parent class (`src/mahjong_wrapper.py:104`).
- `_ensure_gui()` initializes pygame, fonts, the display surface, a `Clock`, and preloads tile assets. It runs on construction and after display rebuilds (`src/mahjong_wrapper.py:241`).
- The main render loop lives in `_render()`. Each frame clears the screen, recomputes layout state, draws every region, flips the display, and ticks the FPS cap (`src/mahjong_wrapper.py:1330`).

## 2. Visual Design System
- Palette constants define background `(12, 30, 60)`, play field `(24, 60, 90)` with border `(40, 90, 130)`, deep gray panels, pale typography, cyan accents, and red danger tint (`src/mahjong_wrapper.py:138`).
- Fonts: base font size comes from `font_size` (default 12). `_ensure_gui()` builds three handles—regular, small (`max(12, base-4)`), and header (`base+8`) (`src/mahjong_wrapper.py:241`).
- Fallback font stack prioritizes CJK-capable faces (various Noto/Source Han/Microsoft families) to guarantee tile labels and seat names render (`src/mahjong_wrapper.py:118`).

## 3. Asset Pipeline & Tile Surfaces
- Tile SVGs are loaded from `assets/tiles/Regular`; each gets composited over a rounded white rectangle for consistent framing (`src/mahjong_wrapper.py:349`).
- Surfaces are cached per tile index and size. Missing SVGs fall back to programmatic placeholders colored by suit with text labels (`src/mahjong_wrapper.py:376`).
- Face-down tiles use a dark rounded rectangle with a lighter outline. Rotated versions (90°, 180°, 270°) are memoized in `_face_down_cache` (`src/mahjong_wrapper.py:453`).
- Tile orientation and scaling honour requested width/height, with rotation caching handled via `_tile_orientation_cache` (`src/mahjong_wrapper.py:408`).

## 4. Layout Geometry & Scaling
- `_render()` defines a square `play_rect` whose side equals the current window height and centers it horizontally at the top of the screen (`src/mahjong_wrapper.py:1354`).
- `_compute_tile_metrics()` sets a base tile width from `play_rect` (bounded 16–72 pixels) and keeps a 1:1.4 aspect ratio. The same size feeds hand, discard, meld, wall, and dead-wall tiles (`src/mahjong_wrapper.py:562`).
- Everything outside `play_rect` (status text, buttons) adapts to the full window width/height.

## 5. Player Zones
- `_draw_player_areas()` creates one transparent surface per player, draws that seat flat, then rotates it by seat angle (South 0°, East −90°, North 180°, West 90°) before blitting to the table (`src/mahjong_wrapper.py:828`).
- `_draw_player_layout()` contents:
  - **Hand**: tiles centered near the bottom of the seat surface, spaced by `tile_width + 6`. Last draw may shift slightly to imply the 14th tile (`src/mahjong_wrapper.py:744`).
  - **Concealment**: `face_up_hand` controls whether tiles render face-up or as the face-down skin. Only South is always visible; others reveal conditionally (see Section 8).
  - **Riichi banner**: cyan “Riichi” badge appears above the hand when that player is in riichi (`src/mahjong_wrapper.py:781`).
  - **Discards**: 6-column grid (max 4 rows) centered above the hand. Declared riichi tile rotates 90° using `orientation_map` (`src/mahjong_wrapper.py:792`).
  - **Melds**: aligned to the right margin, drawn horizontally with claimed tiles sideways and concealed-kan logic flipping end tiles face-down (`src/mahjong_wrapper.py:684`).

## 6. Center Panel & Court Elements
- `_draw_center_panel()` produces a rounded rectangle sized to ~24% of `play_rect`, centered on the table. It shows round wind/hand, honba, riichi stick count, and deck size (`src/mahjong_wrapper.py:584`).
- Player scores appear around the panel at cardinal directions. Current player score adopts the accent color (`src/mahjong_wrapper.py:633`).
- `_draw_dead_wall()` stacks five tiles below the panel, showing live dora indicators on top of the stack (`src/mahjong_wrapper.py:891`).
- Seat labels render just outside each table edge with muted text; the dealer (oya) gains a cyan ring plus an “E” glyph offset toward the table (`src/mahjong_wrapper.py:910`).

## 7. Status Band & Control Buttons
- `_draw_status_text()` prints phase, current player, last action, reward, and an “Episode finished” notice when `done` is true. Positioned in opposing corners of the top margin (`src/mahjong_wrapper.py:951`).
- `_draw_control_buttons()` lays out three buttons stacked at the bottom-right corner: `Pause on Score`, `Auto Next`, and `Next`. Minimum width 160 px, height 44 px, with 10 px gaps (`src/mahjong_wrapper.py:973`).
- Button visuals: active state brightens fill and accent border; disabled state darkens base and mutes text. Rectangles are stored for hit-testing (`src/mahjong_wrapper.py:1001`).

## 8. Score Phase Overlay & Reveal Logic
- `_score_last()` flags the moment when the score phase begins and the last player has acted (`src/mahjong_wrapper.py:343`).
- `_compute_hand_reveal_flags()` exposes winning or tenpai hands in the score phase while keeping others hidden. Outside scoring, only South (player index 0) is face-up (`src/mahjong_wrapper.py:541`).
- `_draw_score_panel()` replaces the status band when `score` is active. A square panel centered in the window shows round metadata, winner/loser messaging, fu/han totals, per-player ranks, point deltas, and wrapped yaku lines (`src/mahjong_wrapper.py:1056`).
- Winners render in accent colors, score deltas colourize by sign, and yaku text aligns left with optional label prefixes (`src/mahjong_wrapper.py:1192`).

## 9. Interaction & Mode Switching
- `_process_events()` handles window close (`QUIT`), resize (recreates display surface with new size), and left-click dispatch to `_handle_mouse_click()` (`src/mahjong_wrapper.py:297`).
- `_handle_mouse_click()` toggles `Auto Next`, enqueues single-step requests, and flips `Pause on Score` when the respective button rect contains the cursor (`src/mahjong_wrapper.py:310`).
- Auto-play pauses when score panels appear if `Pause on Score` is armed. Manual mode requires the `Next` button to continue stepping (`src/mahjong_wrapper.py:320`).
- `_update_riichi_state()` tracks when players declare riichi, storing the discard index for the sideways tile highlight and resetting when states revert (`src/mahjong_wrapper.py:475`).

## 10. Rendering Flow Summary
1. Process events (quit, resize, clicks). (`reset`/`step` call `_process_events()` before touching game state.)
2. Update environment through base class `step`, storing last action/reward/done in `_RenderPayload` (`src/mahjong_wrapper.py:198`).
3. Call `_render()` to refresh the window using the sections outlined above.
4. Flip the display and tick the clock.

## 11. Recreation Checklist
1. Initialize pygame + fonts, create a resizable window, and keep a Clock in sync with your target FPS.
2. Define the color palette, font sizes, and text styles exactly as noted.
3. Implement tile asset loading with caching, placeholder creation, and orientation-aware surfaces.
4. Derive a square play area, compute tile metrics from it, and render one seat per cardinal direction using rotated surfaces.
5. Add the center panel, dead wall, seat labels, status overlay, control button stack, and score overlay logic.
6. Mirror the event handling semantics (quit, resize, buttons) to maintain feature parity.
7. Wrap each render pass with display flip and frame tick to honour the FPS cap.

Following these notes should let you rebuild an identical Mahjong GUI on top of the existing environment mechanics.
