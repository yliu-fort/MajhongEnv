"""Asset management utilities for Mahjong tile textures."""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from kivy.core.image import Image as CoreImage

try:  # pragma: no cover - optional dependency
    from cairosvg import svg2png
except Exception:  # pragma: no cover - optional dependency
    svg2png = None  # type: ignore[assignment]


BackgroundColor = Union[str, Sequence[int], Sequence[float]]


@dataclass
class TileAssetManager:
    """Load and cache Kivy textures for Mahjong tiles."""

    asset_root: Path
    tile_symbols: Sequence[str]
    background_color: Optional[BackgroundColor] = None
    svg_converter: Optional[Callable[..., bytes]] = None
    _textures: Dict[int, Any] = field(default_factory=dict, init=False)
    _current_size: Optional[Tuple[int, int]] = field(default=None, init=False)

    def load(self, target_size: Optional[Tuple[int, int]]) -> None:
        """Load textures for the configured tiles at the requested size."""
        sanitized_size = None
        if target_size is not None:
            width, height = target_size
            sanitized_size = (max(1, int(width)), max(1, int(height)))
        if sanitized_size == self._current_size and self._textures:
            return

        self._textures.clear()
        self._current_size = sanitized_size
        if not self.asset_root.exists():
            return

        converter = self.svg_converter or svg2png
        for tile_index, symbol in enumerate(self.tile_symbols):
            path = self.asset_root / f"{symbol}.svg"
            if not path.exists():
                continue
            texture = self._load_svg(path, converter, sanitized_size)
            if texture is None:
                texture = self._load_fallback(path)
            if texture is not None:
                self._textures[tile_index] = texture

    def _load_svg(
        self,
        path: Path,
        converter: Optional[Callable[..., bytes]],
        target_size: Optional[Tuple[int, int]],
    ) -> Any:
        if converter is None:
            return None
        svg_kwargs: Dict[str, Any] = {"url": str(path)}
        if target_size is not None:
            svg_kwargs["output_width"], svg_kwargs["output_height"] = target_size
        if self.background_color is not None:
            svg_kwargs["background_color"] = self.background_color
        try:
            png_bytes = converter(**svg_kwargs)
        except Exception:
            return None
        try:
            buffer = BytesIO(png_bytes)
            image = CoreImage(buffer, ext="png")
            return image.texture
        except Exception:
            return None

    def _load_fallback(self, path: Path) -> Any:
        try:
            image = CoreImage(str(path))
        except Exception:
            return None
        return image.texture

    def get_texture(self, tile_index: int) -> Any:
        """Return the cached texture for ``tile_index`` if available."""
        return self._textures.get(tile_index)

    @property
    def current_size(self) -> Optional[Tuple[int, int]]:
        return self._current_size


__all__ = ["TileAssetManager"]
