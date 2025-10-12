"""Kivy demo app showcasing runtime language switching with font fallbacks.

The demo mirrors the font fallback strategy used in :mod:`mahjong_wrapper`
so that non-Latin characters (Simplified Chinese and Japanese) render
reliably when the selected language changes at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

from kivy.app import App
from kivy.core.text import LabelBase
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

try:  # pragma: no cover - optional dependency
    import pygame
    import pygame.font
except Exception:  # pragma: no cover - fall back to default fonts when pygame missing
    pygame = None

_FALLBACK_FONTS: tuple[str, ...] = (
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Source Han Sans CN",
    "Source Han Sans TW",
    "Source Han Sans JP",
    "Microsoft YaHei",
    "Microsoft JhengHei",
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "SimHei",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)


def _normalize_font_names(font_name: str | Iterable[str]) -> list[str]:
    """Split a font string into individual names.

    The helper mirrors :meth:`mahjong_wrapper.MahjongEnvGUIWrapper._normalize_font_names`.
    """

    if isinstance(font_name, str):
        return [name.strip() for name in font_name.replace(";", ",").split(",") if name.strip()]
    return [name for name in font_name if isinstance(name, str) and name]


def _match_font(primary: Optional[str] = None) -> Optional[str]:
    """Return the first available font path that can render CJK characters."""

    candidate_names: list[str] = []
    if primary:
        candidate_names.extend(_normalize_font_names(primary))
    candidate_names.extend(name for name in _FALLBACK_FONTS if name)

    if pygame is None or pygame.font is None:
        return None

    if not pygame.get_init():  # pragma: no cover - light safeguard
        pygame.init()
    if not pygame.font.get_init():  # pragma: no cover - light safeguard
        pygame.font.init()

    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        try:
            matched = pygame.font.match_font(name)
        except Exception:  # pragma: no cover - pygame edge case
            matched = None
        if matched:
            # ``match_font`` returns an absolute path on success.
            font_path = Path(matched)
            if font_path.exists():
                return str(font_path)
    return None


def _register_font(font_path: Optional[str]) -> None:
    """Register a font with Kivy if one is available."""

    if font_path:
        LabelBase.register("MultiLangFont", fn_regular=font_path)
    else:
        # ``Label`` will fall back to Kivy's default if registration fails.
        LabelBase.register("MultiLangFont", fn_regular=LabelBase.default_font)


class LanguageSwitcher(BoxLayout):
    """Container that hosts the language selection UI."""

    current_language = StringProperty("en")
    message = StringProperty("Hello, Mahjong fans!")

    _MESSAGES: Mapping[str, str] = {
        "en": "Hello, Mahjong fans!",
        "zh": "\u9ebb\u5c06\u7231\u597d\u8005\uff0c\u4f60\u597d\uff01",
        "ja": "\u30de\u30fc\u30b8\u30e3\u30f3\u30d5\u30a1\u30f3\u306e\u307f\u306a\u3055\u3093\u3001\u3053\u3093\u306b\u3061\u306f\uff01",
    }

    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=16, spacing=12, **kwargs)
        self._build_ui()

    def _build_ui(self) -> None:
        self.label = Label(
            text=self.message,
            font_size="32sp",
            font_name="MultiLangFont",
            halign="center",
            valign="middle",
        )
        self.label.bind(size=self._update_label_text_size)
        self.add_widget(self.label)

        button_row = BoxLayout(orientation="horizontal", size_hint_y=None, height="48dp", spacing=8)
        button_row.add_widget(self._make_button("English", "en"))
        button_row.add_widget(self._make_button("简体中文", "zh"))
        button_row.add_widget(self._make_button("日本語", "ja"))
        self.add_widget(button_row)

    def _make_button(self, label: str, language: str) -> Button:
        button = Button(text=label, font_name="MultiLangFont")
        button.bind(on_release=lambda *_: self.set_language(language))
        return button

    def set_language(self, language: str) -> None:
        if language not in self._MESSAGES:
            return
        self.current_language = language
        self.message = self._MESSAGES[language]
        self.label.text = self.message

    def _update_label_text_size(self, instance: Label, _value) -> None:
        instance.text_size = (instance.width, None)


class LanguageSwitcherApp(App):
    """Kivy application that demonstrates runtime language switching."""

    def __init__(self, font_name: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._font_name = font_name

    def build(self):  # pragma: no cover - UI construction
        font_path = _match_font(self._font_name)
        _register_font(font_path)
        return LanguageSwitcher()


if __name__ == "__main__":  # pragma: no cover - manual execution guard
    LanguageSwitcherApp().run()
