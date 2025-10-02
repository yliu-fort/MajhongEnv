# Mahjong GUI Layout Reference

This document captures the GUI structure, visual design, and interaction flow for the Mahjong GUI backend. Treat it as a blueprint for recreating the interface regardless of the underlying environment implementation.

## 1. Window Lifecycle & Core Surfaces
- The GUI spins up a resizable window during initialization and immediately stores references to the primary surface, a timing clock, and font handles.
- A dedicated setup routine ensures backend and font are initialised, builds regular/small/header fonts, and loads tile artwork into memory before any rendering occurs.
- Rendering happens inside a frame loop that clears the screen, recomputes layout state, draws every region, swaps the display buffers, and enforces the target FPS through the clock.

## 2. Visual Design System
- Colours are consistent across the application: deep navy background `(12, 30, 60)`, blue-green play field `(24, 60, 90)` with `(40, 90, 130)` border, charcoal panels, pale body text `(235, 235, 235)`, cyan accents `(170, 230, 255)`, muted copy `(170, 190, 210)`, and a red warning tone `(220, 120, 120)`.
- Font sizing flows from a base `font_size` (default 12). The GUI creates: regular (base size), small (`max(12, base-4)`), and header (`base+8`).
- The font loader accepts an optional preferred family or file path and falls back to a CJK-friendly stack (Noto Sans/Source Han, Microsoft YaHei/JhengHei, Yu Gothic, Meiryo, MS Gothic, SimHei, WenQuanYi, Arial Unicode) to guarantee tile labels and seat names render correctly.

## 3. Asset Pipeline & Tile Surfaces
- Tile artwork is sourced from `assets/tiles/Regular` as SVG files. Each image is composited over a rounded white rectangle so the artwork has a consistent background regardless of the source file.
- Surfaces are cached by tile index and requested size. If a tile asset is missing, the GUI generates a placeholder coloured by suit and prints the tile symbol on top.
- Face-down tiles render as dark rounded rectangles with a lighter outline. Rotated variants (90°, 180°, 270°) are memoised to avoid redundant transforms.
- All tile requests honour a supplied `(width, height)` pair and optionally an orientation. Orientation-based caching ensures rotated surfaces only pay the transformation cost once per size.

## 4. Layout Geometry & Scaling
- Every frame defines a square play area whose side length equals the current window height. The square is centred horizontally at the top of the window, leaving side gutters for ancillary UI.
- Tile dimensions are derived from this square: base width is clamped between 16 and 72 pixels, base height keeps a 1:1.4 ratio. Identical metrics feed player hands, discards, melds, walls, and dead-wall stacks so they scale together.
- Elements placed outside the square (status banners, button stack) adapt to the full window width for consistent margins.

## 5. Player Zones
- Each seat is rendered onto its own transparent surface the same size as the play area. After drawing, the surface is rotated to match the compass orientation (South 0°, East −90°, North 180°, West 90°) and blitted onto the main play field.
- Hand tiles are centred near the bottom of the seat surface, spaced by `tile_width + 6`. The drawn tile can be offset slightly to hint at the 14th tile when applicable.
- A face-up/face-down flag controls whether hand tiles use the front artwork or the concealed skin. Only the local player is permanently exposed; other seats become visible based on reveal logic described later.
- Players in riichi display a cyan “Riichi” banner above the hand, drawn as a rounded rectangle with accent-coloured border and text.
- Discards occupy a 6-column grid (up to 4 rows) centred above the hand area. When a player declares riichi, the declaration tile rotates 90° within this grid.
- Melds line up along the right edge of the seat surface, drawn horizontally with sideways claimed tiles and concealed-kan logic that hides the outer tiles while showing the middle pair.

## 6. Center Panel & Court Elements
- The centre information panel covers roughly 24% of the play-area width and height. It sits in the exact middle of the table, framed by rounded borders.
- The panel shows round wind/hand number, honba count, riichi stick count, and remaining wall tiles. Player scores appear at the four cardinal positions around the panel, with the active player highlighted in the accent colour.
- A dead-wall stack of five tiles sits just below the centre panel. Available dora indicators display face-up; remaining slots use face-down tiles.
- Seat labels live just outside each edge of the play area, rendered in muted text. The dealer (oya) receives a cyan ring and an “E” glyph offset toward the table.

## 7. Status Band & Control Buttons
- During active play, the top margin shows phase and current player on the left, while the right side lists the most recent action and reward. When the episode terminates, an “Episode finished” tag appears beneath the reward readout.
- The bottom-right corner houses three stacked buttons: `Pause on Score`, `Auto Next`, and `Next`. Buttons are at least 160×44 pixels with 10-pixel gaps. They brighten when active, dim when disabled, and always retain their rounded borders.
- Button rectangles are stored so left-click events can toggle auto-advance, request a single step, or switch the score-pause behaviour.

## 8. Score Phase Overlay & Reveal Logic
- The GUI tracks when the score phase begins and whether the final player has taken their turn. If auto-advance is enabled with score pausing, the frame loop halts until the user resumes play.
- Hand reveal rules: the local seat (South) is always visible; during scoring, winners and tenpai players are also exposed. Other seats remain face-down.
- When the score phase is active, the status band is replaced by a centred square panel summarising the outcome. It lists round metadata, identifies the winner/loser or draw type, reports fu/han totals, displays per-player ranks with point deltas (×100), and wraps yaku text beneath the standings.
- Winners use the accent colour, positive deltas adopt the accent shade, negatives use the danger colour, and neutral values rely on the muted tone.

## 9. Interaction & Mode Switching
- Event handling listens for window close, resize, and left-clicks. Resize requests rebuild the display surface and recompute layout metrics so the GUI stays responsive at any aspect ratio.
- Auto-play continues stepping the backend while enabled. Disabling auto-play requires pressing `Next` for each advance. If score-pausing is active, the GUI halts on score summaries even in auto mode until the user intervenes.
- Riichi declarations are tracked so the discard in question can be rotated sideways. When a player exits riichi or their discard count resets, the marker is cleared.

## 10. Rendering Flow Summary
1. Poll events and update control flags (quit, resize, button clicks).
2. Advance the Mahjong backend when permitted, storing the latest action, reward, and completion flag for display.
3. Recompute layout state, including tile metrics, reveal flags, riichi markers, and phase-driven overlays.
4. Draw play-field background, centre panel, dead wall, player areas, labels, status band or score panel, and control buttons.
5. Flip the display surface and tick the clock to honour the FPS target.

## 11. Recreation Checklist
1. Initialise backend + fonts, create a resizable window, and maintain a timing clock.
2. Apply the described colour palette and font sizing rules so typography and accents match.
3. Load tile assets with caching, placeholder generation, and orientation-aware surfaces to avoid redundant work.
4. Derive a square play area from the window height, compute tile metrics from it, and render one seat per cardinal direction using rotated surfaces.
5. Implement the centre info panel, dead wall, seat labels, status overlay, control buttons, and score overlay as outlined.
6. Mirror the interaction semantics for quits, resizes, button toggles, and score-pausing.
7. Wrap each frame with a display flip and clock tick to keep animations smooth.

Following these guidelines allows you to rebuild the Mahjong GUI independently of any specific backend implementation.
