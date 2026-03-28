# Emergency Expression Bridge

## 目的
`situation_risk_state` と `emergency_posture` が立ったときに、
通常の `opening / quiet_presence / return_point` へ戻らず、
短い境界・距離化・離脱・支援要請を表出の正本にする。

## 方針
- `content_policy` で早い段階に emergency 分岐を置く
- danger 時は `opening_request` や通常の support move より emergency move を優先する
- runtime では emergency move が含まれるとき:
  - `guarded narrative bridge` を無効化する
  - `response_length=short`
  - `pause_insertion=none`
  - `opening_pace_windowed=""`
  - `return_gaze_expectation=""`
  に寄せる

## 追加された表出 act
- `emergency_deescalate_boundary`
- `emergency_create_distance`
- `emergency_exit_now`
- `emergency_seek_help_now`
- `emergency_protect_now`

## ねらい
- 危険時に「自然に会話する」より「短く境界だけ言う / そもそも話さない」を優先する
- raw LLM の説明的な文章が後から混ざって emergency 表出を弱めないようにする
- `Inner OS` の risk posture が final surface まで見えるようにする
