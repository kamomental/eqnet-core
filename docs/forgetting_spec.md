EQNet Nightly Forgetting & Defrag Specification
(疎結合・最小構成・Monument前提)

1. 目的 (Why)
本仕様は、EQNetにおける「忘却」を
記憶の削除ではなく想起経路(参照重み w)の再配分として定義する。

目的は最適化ではなく、
物語の連続性・関係性の安定・人格のねじれ回避・説明可能性の確保である。

2. 適用範囲 (Scope)
- 実行タイミング: Nightlyのみ
- 日中の対話・判断ループでは w を一切変更しない

対象
- 記録 (content / phi / psi): 変更しない
- 想起経路の重み (w): Nightlyでのみ更新

非対象 (明示)
- 記憶の削除
- 即時リセット
- 日中の人格変動

3. 忘却モデル (3因子)
Nightlyのw更新は、以下3系統の入力証拠に基づく。

A. Recall Evidence
- 直近N日の想起回数
- 使われない記憶は想起されにくくなる
- 反復想起は保護される (spacing効果)

B. Affect Load
- self_reportのstress / pain / confidence等
- 想起後に状態が悪化したか
- Reconsolidation判断の材料

C. Interference Evidence
- phi距離の近傍密度
- 類似記憶が増えるほど古い記憶は押し出される
- 主にL1/L2に適用

4. Monument前提のガード (必須)
4.1 Monumentの定義
- Monumentは昇格した記念碑的記憶
- 物語・関係性・自己同一性の核

4.2 Monument自体の扱い
- Monumentのwは固定 (更新対象外)
- 設定: forgetting.monument_w_lock: true

4.3 Monument接続維持 (floor)
- Monument自体は動かさない
- ただしMonumentへ到達する周辺L1/L2経路は下限(floor)で維持
- 「核だけ残って辿り着けない」状態を防ぐ
- 設定: forgetting.monument_connection_floor
※ floorは「Monumentが想起可能であること」を保証する参照確率の下限であり、
  記憶の重要度(w)そのものを保存する機構ではない。

5. w更新ポリシー (順序固定)
w更新は必ず以下の順序で適用する (順序は変更不可)。

1) consent_override
2) anchor / monument_guard
3) 3因子合成 (Recall / Affect / Interference)
4) 監査出力

設定: forgetting.policy_order
※係数調整は可、順序は不可。

6. Reconsolidation (再固定化)
- Reconsolidationの主成果物は narrative / self_report の更新
- w更新は副作用として緩やかに行う (急変禁止)

設定:
- forgetting.reconsolidation_rate
- forgetting.max_delta_w

意図:
「意味が変わったから影響が薄れる」を再現する。

7. 入力ソース (流入経路の固定)
Nightlyのw更新入力は、以下を同等に扱う。

- Hub系ログ
- HeartOSセッションランナー由来ログ
  (self_report / moment / narrative / pdc 等)

7.1 ソース管理
- すべての入力はソースタグ付き
- 監査ログで出所を追跡可能

設定: forgetting.input_sources

7.2 ソース重み
- ソース差の重み付けは設定で制御
- デフォルトは同重み

設定: forgetting.source_weights

8. 監査・説明 (必須)
8.1 個別ログ (JSONL)
- 対象記憶ID
- w before / after
- 寄与因子 (Recall / Affect / Interference / Consent / Guard)
- クリップ / floor発動有無
- 入力ソースタグ

8.2 集計ログ (Nightly Summary)
- w変化量分布
- Monument接続floor発動回数
- 同意撤回による抑制件数
- 過剰変化検知用指標

設定:
- forgetting.audit.enable_item_log
- forgetting.audit.enable_summary

9. Defrag (構造整備) ※Forgettingと独立
9.1 定義
Defragは、記録を削除せず参照コストと矛盾を減らし、
物語の整合性を上げるNightly処理。

9.2 Forgettingとの関係
- Forgetting (w再配分)とは独立したNightlyステップ
- 目的と出力が異なるため疎結合で共存
- ForgettingとDefragは同一Nightly内で実行されうるが、互いの結果を前提条件としない

9.3 Defragの操作 (例)
- 重複統合 (Dedup / Merge)
- 参照索引の再構築 (Reindex)
- 競合の分離 (Conflict Isolation)
- ナラティブ圧縮 (Narrative Rollup)

9.4 Monumentとの整合
- DefragはMonumentを変更しない
- Monumentへの参照経路の整備は許可

設定 (任意):
- defrag.enable
- defrag.cluster_similarity_threshold
- defrag.conflict_isolation_strength
- defrag.rollup_trigger_budget

10. 非目標 (明示)
本仕様は以下を行わない。
- 記憶の物理削除
- 日中のw更新
- Monumentの降格・再評価
- 学習・最適化ループへの介入

11. 設計の芯 (要約)
- 忘却は削除ではなく再配分
- 操作対象はwのみ
- 実行はNightly限定
- Monumentは固定+接続維持
- 3因子入力で人間らしい忘却
- 監査・説明が前提

説明レイヤ指針は運用側のRunbookに集約する。
詳しくは docs/forgetting_runbook.md を参照する。

関連ドキュメント
- docs/forgetting_runbook.md (運用チェックリスト)
