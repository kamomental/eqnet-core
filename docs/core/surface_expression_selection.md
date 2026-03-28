# Surface Expression Selection

`surface_expression_selection` は、表出を単純な禁止語リストや固定文分岐で制御しないための最小基盤です。

## 方針

- `inner_os` は act と surface intent までを決める
- 最終文面は、その act に対応する候補群から選ぶ
- 選択は `register / warmth / intimacy / genericity / consultation_tone / brevity` の属性で行う

つまり、

- 常時禁止
- 条件付き回避
- 条件付き許容
- 条件付き推奨

を、phrase ごとの属性と profile の相性で扱います。

## 目的

- generic fallback の抑制
- 相談窓口テンプレへの退避抑制
- キャラクター整合に応じた口調選択
- `live mode` や `love mode` を別人格化せず modulation として扱うこと

## 現在の適用範囲

まずは compact phrase の選択点だけに適用しています。

- `thread_reopen_return`
- `deep_reflection_stay`
- `deep_reflection_presence`
- `continuity_opening`
- `thread_presence`

今後は `reflect_only / quiet_presence / light question / reopen_from_anchor` の phrasebank 全体へ広げます。
