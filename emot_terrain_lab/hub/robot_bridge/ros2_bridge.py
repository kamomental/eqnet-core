# -*- coding: utf-8 -*-
"""
ROS2 robot bridge (mock implementation).

This scaffold does not require rclpy at runtime; instead it logs the controls
that would be sent to ROS2 topics. Real deployments can replace the methods
with actual ROS2 publishers.
"""

from __future__ import annotations

import time
from typing import Dict

try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    HAS_ROS2 = True
except Exception:
    HAS_ROS2 = False
    Node = object

from .base import BaseRobotBridge, RobotBridgeConfig


class ROS2Bridge(BaseRobotBridge):
    """Mock ROS2 bridge."""

    def __init__(self, config: RobotBridgeConfig) -> None:
        super().__init__(config)
        self._node: Node | None = None

    def connect(self) -> None:
        if not self.config.enabled:
            return
        if HAS_ROS2:
            rclpy.init()
            self._node = rclpy.create_node("eqnet_robot_bridge")
            self.state.status = "connected"
        else:
            self.state.status = "mock_connected"

    def publish(self, controls: Dict[str, float]) -> None:
        if not self.config.enabled:
            return
        self.state.last_controls = controls
        self.state.last_publish_timestamp = time.time()
        # In mock mode we simply print; production code would publish geometry/Twist etc.
        if HAS_ROS2 and self._node:
            # TODO: Implement real publishers (Twist, trajectory, etc.)
            self.state.status = "published"
        else:
            print("[ROS2Bridge mock] controls:", controls)
            self.state.status = "mock_published"

    def close(self) -> None:
        if HAS_ROS2 and self._node:
            self._node.destroy_node()
            rclpy.shutdown()
        self.state.status = "closed"
