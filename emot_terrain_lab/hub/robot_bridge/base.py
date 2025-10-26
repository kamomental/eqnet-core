# -*- coding: utf-8 -*-
"""
Base interfaces for robot bridges.

A robot bridge is responsible for converting affect controls (coming from the
policy head) into physical actions—velocity, gestures, gaze, etc.  Different
robot backends (ROS2, Isaac Sim, custom hardware) should inherit from the base
class here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class RobotBridgeConfig:
    """Common robot bridge configuration."""

    kind: str = "ros2"  # e.g., "ros2", "isaac", "mock"
    enabled: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotState:
    """Diagnostics / state to expose from bridges."""

    last_publish_timestamp: Optional[float] = None
    last_controls: Dict[str, float] = field(default_factory=dict)
    status: str = "idle"


class BaseRobotBridge:
    """Abstract bridge interface."""

    def __init__(self, config: RobotBridgeConfig) -> None:
        self.config = config
        self.state = RobotState()

    def connect(self) -> None:
        """Connect to the robot backend (if necessary)."""
        raise NotImplementedError

    def publish(self, controls: Dict[str, float]) -> None:
        """Publish control signals to the robot."""
        raise NotImplementedError

    def close(self) -> None:
        """Clean up the bridge."""
        raise NotImplementedError
