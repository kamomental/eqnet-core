# Qualia Membrane Projection Model

Date: 2026-03-18

## 目的

この文書は、`クオリア膜` そのものの作用だけを切り出して説明する。

ここで扱うのは次である。

- 膜が何を受け取るか
- 膜が何を射影するか
- 膜が何を落とし、何を通すか
- 膜が人間工学的に何を守るか

`contact point` から `fastpath / midpath / slowpath` までを含む全体像は、
別文書の [qualia_contact_access_model.md](/C:/Users/kouic/Desktop/python_work2/emotional_dft/docs/core/qualia_contact_access_model.md)
で扱う。

## 基本位置づけ

`クオリア膜` は「全部を見せる膜」ではない。

`クオリア膜` は、
**高次元の内部場から、アクセス可能で、かつ接触可能な低次元表現へ落とす射影層**
である。

ここでいう内部場には、少なくとも次が含まれる。

- physiology
- affective terrain
- lingering affect
- relation loading
- culture / norm pressure
- memory ignition
- scene pressure

膜はこれらをそのまま public にしない。
代わりに、

- 圧縮
- 統合
- 前景化
- 可報告化準備
- 接触可能性の整形

を行う。

## 数学的な見方

内部状態を \(x(t)\)、接触点からの局所展開を \(z_i\) とすると、
膜は次のような射影として置ける。

\[
\pi_Q : \mathcal{Z} \to \mathcal{A}
\]

- \(\mathcal{Z}\): 接触点から展開された高次元表現
- \(\mathcal{A}\): access space

実際には、全状態を直接射影するより、
各接触点からの写像結果を重み付きで束ねて射影する方が自然である。

\[
a_{\mathrm{access}}(t)=\pi_Q\Bigl(\sum_i w_i z_i\Bigr)
\]

ここで \(w_i\) は次に依存する。

- terrain
- relation
- scene
- temporal response
- safety / boundary pressure

つまり膜は、
**何が触れたか** だけでなく、
**今それをどこまで通してよいか**
を見ている。

## 膜がやる仕事

### 1. Compression

高次元の内的状態を、そのままでは扱えない。
膜はまずそれを、前景候補として扱える形に圧縮する。

### 2. Integration

複数の contact point や複数の内的張力を束ねる。
この段階では、まだ完全に言語化されない。

### 3. Foregrounding

何が「いま触れているか」を前景化する。
ただし、前景化されたもの全てが可報告になるわけではない。

### 4. Reportability Preparation

膜は reportable slice そのものではない。
膜は、その準備をする。

### 5. Ergonomic Admissibility

これが今回の追加で最も重要である。

膜は単に access を作るだけでなく、
**人間工学的に通してよい形へ整える**。

たとえば膜は、次を抑制または整形する。

- 過負荷を増幅する露出
- 境界を破る disclosure
- 修復不能な踏み込み
- public scene での過剰接近
- 未整理なままの過剰言語化

## 人間工学から見た膜の役割

`クオリア膜` を人間工学的に言い換えると、
**接触を成立させる前の安全・距離・可視性の調整層**
である。

したがって膜の評価は、
「たくさん前景化できたか」ではなく、次で見るべきである。

- overload を増やさなかったか
- disclosure overshoot を防げたか
- repair 余地を残せたか
- dignity を壊さなかったか
- same-scene / same-partner の文脈に合っていたか

## 膜と Conscious Access の違い

`conscious access` と `クオリア膜` は重なるが同一ではない。

- 膜:
  - 高次元場を access 可能な形へ落とす
  - 接触可能性を整える
  - 前景化の前段を担う

- conscious access:
  - 実際に何が foreground に上がるか
  - 何が行為や言語に渡るか
  - 何が withheld されるか

つまり、
膜は `access possibility shaping`、
conscious access は `actual foreground control`
である。

## 膜と Reportable Slice の違い

`reportable slice` は public 側に近い。
膜はそのもっと手前にある。

- 膜:
  - まだ前景候補の整形段階
- reportable slice:
  - 実際に発話・行為・memory write に出る切片

この区別がないと、
「内面で触れたこと」と「外に出したこと」が混ざる。

## 膜と fastpath の関係

膜は fastpath を全面的に経由しない。
ただし fastpath と無関係でもない。

自然なのは次の分担である。

- fastpath:
  - body / boundary / affordance の最短経路
  - 保護、抑制、即時回避

- membrane projection:
  - access へ通す前の整形
  - 前景化してよい形への調整

したがって、fastpath は膜の代わりではなく、
**膜が十分に働く前に保護を優先する最短経路**
である。

## 膜と scene / relation / culture

膜は pure な内面装置ではない。
常に scene と relation によって変形される。

同じ contact point でも、

- public
- private
- repair 中
- reverent distance
- shared work

で、膜の通し方は変わる。

つまり膜射影は、

\[
a_{\mathrm{access}}(t)=\pi_Q\bigl(z, scene, relation, culture, safety\bigr)
\]

と考えた方が正確である。

## 工学的まとめ

膜単体の作用を工学的にまとめると、次になる。

- 入力:
  - contact point からの局所展開
  - terrain / relation / scene / safety
- 処理:
  - compression
  - integration
  - foreground shaping
  - ergonomic admissibility filtering
- 出力:
  - access-ready representation
  - foreground candidate weighting
  - reportability preparation

## 一文サマリ

`クオリア膜` とは、
接触点から立ち上がった多重な内的展開を、
その場の relation・scene・safety に照らして、
**前景化してよい形、かつ接触してよい形へ整える射影層**
である。
