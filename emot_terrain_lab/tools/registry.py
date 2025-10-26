# -*- coding: utf-8 -*-
"""
Skill registry for the EQNet hub.

The registry keeps track of available skills (intent routing) that can be
expanded over time. This is a lightweight layer that the LLM hub can consult.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml

SKILL_CONFIG_PATH = Path("config/tools.yaml")


@dataclass
class Skill:
    id: str
    name: str
    intent: str
    description: str
    llm: str
    context_sources: List[str]
    priority: int = 10


class SkillRegistry:
    """Simple in-memory skill store."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or SKILL_CONFIG_PATH
        self.skills: Dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        if not self.config_path.exists():
            self.skills = {}
            return
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        skills = {}
        for item in data.get("skills", []):
            skill = Skill(
                id=item["id"],
                name=item.get("name", item["id"]),
                intent=item.get("intent", "chitchat"),
                description=item.get("description", ""),
                llm=item.get("llm", "llm-fast"),
                context_sources=item.get("context_sources", []) or [],
                priority=int(item.get("priority", 10)),
            )
            skills[skill.id] = skill
        self.skills = skills

    def find_by_intent(self, intent: str) -> Optional[Skill]:
        """Return the highest-priority skill matching the intent."""
        matches = [skill for skill in self.skills.values() if skill.intent == intent]
        if not matches:
            return None
        matches.sort(key=lambda s: s.priority)
        return matches[0]

    def list_skills(self) -> List[Skill]:
        return sorted(self.skills.values(), key=lambda s: (s.priority, s.id))

