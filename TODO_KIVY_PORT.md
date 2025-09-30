# Mahjong Wrapper Kivy Port Plan

The pygame-based `MahjongEnv` wrapper is large (roughly 1,600 lines) and its
logic interleaves environment bookkeeping with rendering code. Porting it to
Kivy requires a staged effort to ensure feature parity. The work is broken into
subtasks below, each intended to match behaviour one-to-one with the pygame
implementation while adapting it to the Kivy API surface.

## Subtasks

1. **GUI infrastructure parity**
   - Configure and bootstrap the Kivy `EventLoop`, window, and layout hierarchy.
   - Implement `MahjongEnv` control flow hooks (`_process_events`, `_render`,
     button handlers) against Kivy events and widgets.
   - Ensure window resize, quit handling, FPS ticking, and status updates mirror
     pygame semantics.

2. **Tile asset pipeline**
   - Load SVG assets with CairoSVG (with fallback to dynamically rendered
     placeholders when conversion is unavailable), caching per tile + size.
   - Re-implement face-up, face-down, and rotated tile rendering using Kivy
     canvas instructions with correct orientation and scaling logic.

3. **Board composition primitives**
   - Recreate drawing helpers for player areas, melds, discards, walls, center
     panel, score panel, and seat labels using Kivy's canvas API. Maintain all
     layout, spacing, colour, and text calculations from the pygame code.

4. **Dynamic state features**
   - Port riichi tracking, discard rotation markers, score-phase pausing, and
     reveal logic exactly as in pygame version.
   - Reproduce score summary overlays, message rendering, and button state
     management.

5. **Testing & validation**
   - Manual smoke testing to confirm parity with the pygame UI.
   - Automated lint/style adjustments ensuring the module integrates cleanly
     with existing code paths.

## Validation notes

- Added a pytest smoke test (`tests/test_mahjong_wrapper_kivy_import.py`) that
  ensures the module exposes the expected `MahjongEnv` class when Kivy is
  installed, while skipping automatically when the dependency is absent.
- Documented the optional dependency set and usage pattern in the
  [project README](README.md) so that manual validation steps mirror the pygame
  workflow.
- Extended `requirements.txt` with the optional GUI dependencies to simplify
  environment setup prior to running smoke tests.

Completing these subtasks will yield a feature-complete `mahjong_wrapper_kivy.py`
that functions as a drop-in replacement for the pygame wrapper.
