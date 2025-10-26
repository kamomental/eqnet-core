# Robot Bridge (Body) Overview

EQNet ハブの「身体」レイヤーは、感情コントロールをロボットやアバターに伝える役割です。ここでは現状のスケルトンと拡張ポイントを整理します。

---

## 1. ディレクトリ構成
```
hub/robot_bridge/
  base.py        # BaseRobotBridge, RobotBridgeConfig, RobotState
  ros2_bridge.py # ROS2向けブリッジ（モック実装）
  __init__.py
config/robot.yaml # ブリッジ有効化・パラメータ設定
```

`RuntimeConfig` に `robot: RobotBridgeConfig` が追加されており、`enabled: true` でブリッジが起動します。

---

## 2. 使い方
1. `config/robot.yaml` を編集。
   ```yaml
   bridge:
     enabled: true
     kind: "ros2"   # "mock" なら標準出力にログのみ
     params:
       cmd_topic: "/cmd_vel"
       gaze_topic: "/gaze_target"
       publish_hz: 15
   ```
2. ハブ起動時に `EmotionalHubRuntime` が `ROS2Bridge` を生成し、`policy.affect_to_controls` の出力から身体向けコントロール（pause_ms / gesture_amplitude / prosody_energy / gaze_mode）を publish します。
3. ROS2 環境がある場合は、`ros2_bridge.py` に実際の publisher を実装してください（現在はモックで `print` しています）。

---

## 3. 拡張ポイント
- **ROS2 実装**: `rclpy` を用いて `/cmd_vel` 等に Twist を publish、視線やジェスチャは `/servo` などに送る。  
- **Isaac Sim**: `hub/robot_bridge` に `isaac_bridge.py` を追加し、Sim API からアバターを動かす。  
- **センサー入力**: ROS2 の感覚情報を Perception Bridge に流す場合は、専用ノードで `observe_video` / `observe_audio` を呼ぶ。  
- **ロボット状態**: `RobotState` に電池状況や衝突情報などを入れて、EQNet 側の整流（calm90 等）に活かす。

---

## 4. EQNetとの関係
- EQNet Core（心）が `controls` を決め、Robot Bridge（身体）がそれを実世界／シミュレータへ伝えます。  
- 口（LLM/TTS）と同じ段にあり、心 = E → 行為 = controls → 身体/口 の順番を守ることで、スキルが増えても寄り添い挙動が一貫します。
