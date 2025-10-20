from __future__ import annotations

from dataclasses import dataclass

from .visual_agent import VisualAgent


@dataclass(slots=True)
class AgentCard:
    """Metadata bundle for a reusable Mahjong agent instance."""

    identifier: str
    title: str
    model_path: str
    preview_image: str
    agent: VisualAgent
