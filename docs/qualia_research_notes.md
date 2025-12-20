# Qualia Research Notes

A survey of recent academic work that treats qualia as structural, measurable phenomena, providing anchors for EQNet’s virtual DNA / nightly design.

## 1. Programs & References

1. **Qualia Structure Study (Japan, 2023–2028)** – focuses on relational structure of qualia using psychophysics + mathematical models. <https://qualia-structure.jp/intro/research/>
2. **Unconscious processing vs. conscious access** – Continuous Flash Suppression experiments show subjective brightness processed unconsciously. (Consciousness & Cognition, 2025). <https://qualia-structure.jp/news/>
3. **Mathematical phenomenology** – qualia characterized via relational structures/enriched categories (Yoneda-style) in JSPS project publications. <https://ci.nii.ac.jp/>
4. **Predictive error coding / query act** – Frontiers 2023 paper treats qualia as “query acts” arising from predictive coding. <https://www.frontiersin.org/articles/10.3389/fpsyg.2023.1109767>
5. **Modeler Schema Theory** – preprint proposing a schema agent that ignites conscious reports via qualia-based consistency checks. <https://arxiv.org/abs/2307.11111>
6. **Algebraic theory of qualia discrimination** – algebraic independence conditions for qualia spaces. <https://arxiv.org/abs/2401.00513>

## 2. Implementation Skeleton for EQNet

| Module | Research Anchor | Role in EQNet |
| --- | --- | --- |
| **QualiaGraph** | Qualia Structure Project; enriched category models | Maintain similarity/distance matrix `D` for prototypical qualia. Represent each qualia as relation profile `ϕ(q) = [D(i,1)…D(i,N)]` instead of direct contents. Nightly pass updates clusters / “periodic table”. |
| **QueryEngine** | Predictive error coding (Frontiers, 2023) | Drive qualia dynamics via weighted prediction errors `u_t = ||Π_t ε_t||` (precision-weighted). Existing shadow/replay errors supply inputs. |
| **AccessGate** | CFS experiments (unconscious brightness) | Separate processing vs. conscious access with gate `P(a_t=1)=σ(α u_t − β L_t − θ_t)`. Allows recognition without report until gate fires. |
| **MetaMonitor** | Modeler-Schema theory | Add meta-inconsistency term `m_t = Div(WorldModel_t, Post_t)` so access can fire when global schema breaks even if error is small: `σ(αu_t + γm_t − βL_t − θ_t)`. |

### Nightly Integration
1. **QualiaGraph update** – learn/refresh `D`, cluster families (supports “qualia periodic table”).
2. **Gate retuning** – adjust thresholds/precisions (`θ_t`, `Π_t`) to avoid chronic over-arousal (glymphatic analogy).
3. **Telemetry** – expose `u_t`, access state `a_t`, meta term `m_t` for analysis (compare with CFS / schema theory predictions).

This structure lets EQNet treat qualia in terms of relations, dynamics, access gates, and meta-consistency—matching current research without claiming to generate raw qualia.
