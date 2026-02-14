# Replay Asset Layout

`docs/replay/replay.html` が参照する PNGTuber アセットの配置ルールです。

## フォルダ構成

- `assets/replay/src/`
  - 元画像（1枚絵）を置く
  - 例: `assets/replay/src/default.png`
- `assets/replay/masks/<character>/`
  - ガイド画像、手修正マスク、アンカーJSONを置く
  - 例: `assets/replay/masks/default/anchors.json`
- `assets/replay/character/<character>/`
  - ビルド済みレイヤーPNGの出力先
- `assets/replay/pngtuber_manifest.json`
  - `replay.html` が読むマニフェスト

## まず自動生成

```powershell
python tools/replay/build_pngtuber_pack.py --source assets/replay/src/default.png --out-dir assets/replay/character/default --manifest assets/replay/pngtuber_manifest.json
```

出力:
- `assets/replay/character/default/*.png`
- `assets/replay/character/default/build_report.json`
- `assets/replay/masks/default/mask_preview.png`
- `assets/replay/masks/default/anchors.auto.json`

## 3点アンカーで位置を固定

`assets/replay/masks/default/anchors.json` を作り、左目・右目・口の中心を指定します。

```json
{
  "left_eye": { "x": 345, "y": 486 },
  "right_eye": { "x": 505, "y": 486 },
  "mouth": { "x": 425, "y": 742 }
}
```

実行:

```powershell
python tools/replay/build_pngtuber_pack.py --source assets/replay/src/default.png --out-dir assets/replay/character/default --manifest assets/replay/pngtuber_manifest.json --anchors-file assets/replay/masks/default/anchors.json
```

## マスクを手修正して品質を上げる

必要なら `assets/replay/masks/default/` の以下を編集して再実行します。

- `eye_open.png`
- `eye_half.png`
- `mouth_a.png`
- `mouth_i.png`
- `mouth_u.png`
- `mouth_e.png`
- `mouth_o.png`
- `mouth_n.png`

手修正マスク指定例:

```powershell
python tools/replay/build_pngtuber_pack.py --source assets/replay/src/default.png --out-dir assets/replay/character/default --manifest assets/replay/pngtuber_manifest.json --anchors-file assets/replay/masks/default/anchors.json --head-mask assets/replay/masks/default/head.png --body-mask assets/replay/masks/default/body.png --hair-front-mask assets/replay/masks/default/hair_front.png --mouth-a-mask assets/replay/masks/default/mouth_a.png --mouth-i-mask assets/replay/masks/default/mouth_i.png --mouth-u-mask assets/replay/masks/default/mouth_u.png --mouth-e-mask assets/replay/masks/default/mouth_e.png --mouth-o-mask assets/replay/masks/default/mouth_o.png --mouth-n-mask assets/replay/masks/default/mouth_n.png --auto-split false
```
