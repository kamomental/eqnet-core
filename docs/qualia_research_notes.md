# Qualia Research Notes for EQNet

Purpose:
Translate current qualia-related academic research into operational control
structures usable inside EQNet (without attempting to generate qualia itself).

This document assumes:
- Qualia emergence is primitive (not generated here)
- EQNet models propagation, access, and control only

------------------------------------------------------------
1. Academic Anchors (ASCII-safe references)
------------------------------------------------------------

- Qualia Structure Project (Japan, JSPS Transformative Research Area)
  Focus: relational structure of qualia via similarity judgments
  https://qualia-structure.jp

- Predictive Error Coding / Query Act
  Frontiers in Psychology (2025)
  Qualia as dynamic query driven by prediction error

- Conscious Access Dissociation (CFS / illusion studies)
  Consciousness and Cognition / Nature Neuroscience
  Evidence that processing occurs without reportable awareness

- Modeler-Schema Theory (arXiv preprint, 2025)
  Conscious access as schema consistency check (meta-monitor)

------------------------------------------------------------
2. Model Core (Summary)
------------------------------------------------------------

EQNet does NOT model qualia contents.

EQNet DOES model:
- relational structure between qualia (QualiaGraph)
- query pressure from prediction error (QueryEngine)
- access gating (AccessGate)
- schema divergence detection (MetaMonitor)

------------------------------------------------------------
3. State Holders
------------------------------------------------------------

(1) QualiaGraph
    - prototypes q_i
    - distance matrix D[i,j]
    - representation: phi(q_i) = D[i,:]

(2) QueryEngine
    - scalar u_t = || Pi_t * epsilon_t ||

(3) MetaMonitor
    - scalar m_t = divergence(WorldModel_t, PostState_t)

(4) AccessGate
    - computes probability p_t and binary decision a_t

------------------------------------------------------------
4. Gate Equation
------------------------------------------------------------

logit_t = alpha * u_t + gamma * m_t - beta * L_t - theta_t
p_t     = sigmoid(logit_t)
a_t     = 1 if p_t >= tau else 0

Notes:
- L_t is cognitive load / competition
- theta_t is homeostatic threshold (nightly tuned)

------------------------------------------------------------
5. Smoothing / Stability
------------------------------------------------------------

- p_t should be EMA-smoothed to avoid chatter
- hysteresis recommended:
    open_threshold  > close_threshold

------------------------------------------------------------
6. Nightly Duties
------------------------------------------------------------

- QualiaGraph update:
    - recompute D
    - cluster prototypes
    - log cluster metadata ("qualia periodic table")

- Gate retuning:
    - adjust theta_t to keep access rate near target

------------------------------------------------------------
7. Telemetry
------------------------------------------------------------

Log per turn:
- u_t
- m_t
- L_t
- p_t
- a_t
- theta_t
- qualia_cluster_id (if any)

------------------------------------------------------------
8. Implementation Checklist
------------------------------------------------------------

[ ] Add QualiaGraph module
[ ] Compute u_t in inner_replay
[ ] Compute m_t in conscious stack
[ ] Insert AccessGate before narrative
[ ] Log unconscious successes
[ ] Nightly: update graph + retune theta
