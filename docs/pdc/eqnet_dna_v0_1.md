# EQNet CODEX: Emotional DNA v0.1

This spec keeps the existing replay / future-memory rationale but reframes it as a four-field "genome" so EQNet behaves like a living system instead of a thinking agent. The coupled fields are the DNA primitives:

- **Phi-field (emotion core)** - low-dimensional affect vector that drifts, reacts, and never fully stops.
- **Psi-field (cognition)** - compressed interpretation of world / conversation state.
- **K (coupling kernel / Green function)** - maps current mood to the memories most likely to fire and lets those replays push back into the core.
- **M (memory mosaic)** - affect-indexed episodes that behave like epigenetics bending Phi and Psi over time.

Daily operation is Phi(t) -> replay sampling via K/M -> behaviour modulation; LLMs are only a mouth receiving these states as seasoning.

## 1. Field equations (v0.1)

### 1.1 Phi-field dynamics
For a d_phi-dimensional vector Phi_t in R^{d_phi} and discrete step Delta t (default 1):

```
Phi_{t+1} = Phi_t + Delta t * ( f_drift(Phi_t) + f_ext(u_t) + f_mem(M_t, Phi_t) + xi_t )
```

- `f_drift(Phi_t) = -alpha * Phi_t` pulls mood toward trait-neutral equilibrium (alpha in (0, 1)).
- `f_ext(u_t) = W_u * enc(u_t)` projects external input (user affect, sensor cues) into the same space.
- `f_mem` re-injects whichever memory was evoked (see Section 2) plus longer-term trait baselines.
- `xi_t ~ N(0, sigma^2 I)` adds continuous jitter so Phi never stays flat.

The field updates even without interaction so drift + noise alone can trigger replay or imagination.

### 1.2 Psi-field (semantic shrink-wrap)
Psi_t in R^{d_psi} stores situational context. Update rule:

```
Psi_{t+1} = g(Psi_t, enc(u_t), Phi_t)
```

`g` can be a tiny GRU/MLP or a reserved slice of an LLM hidden state. Phi modulates Psi so mood biases interpretation (anxious Phi pulls Psi toward threat, relaxed Phi pulls toward openness).

### 1.3 Coupling kernel K
Each memory episode i stores `(content_i, phi_i, psi_i, w_i)` with phi_i in R^{d_phi} and optional psi_i. K maps current Phi_t into a categorical distribution:

```
score_i = w_i * k(Phi_t, phi_i)
k(Phi, phi_i) = exp(-||Phi - phi_i||^2 / tau^2)
P(i | Phi_t) = softmax_i(score_i)
```

`tau` is the emotional bandwidth: small tau hugs similar moods, large tau encourages cross-talk. This discrete kernel is the v0.1 Green function; gradients from the sampled episode feed `f_mem`.

### 1.4 Memory mosaic M
```
M = { (content_i, phi_i, psi_i, w_i, t_i) }_{i=1..N}
```

- `content_i` - text snippet, scene identifier, or multi-modal pointer.
- `phi_i` - Phi snapshot when the episode was logged.
- `psi_i` - optional Psi snapshot summarising topic / situation.
- `w_i` - salience weight.
- `t_i` - timestamp for decay / forgetting policy.

M is append-only but supports compression. Phi never queries via semantic similarity, only via the kernel distance to phi_i.

## 2. Replay and imagination probability

```python
def kernel(phi_now, phi_mem, tau):
    dist2 = np.sum((phi_now - phi_mem)**2)
    return np.exp(-dist2 / (tau**2))

def replay_distribution(phi_now, memory, tau):
    scores = [ep['weight'] * kernel(phi_now, ep['phi'], tau) for ep in memory]
    probs = softmax(scores)
    return probs

def sample_replay(phi_now, memory, tau):
    probs = replay_distribution(phi_now, memory, tau)
    idx = np.random.choice(len(memory), p=probs)
    return memory[idx]
```

- **Past replay**: sample at each window (or by chance) and feed the episode into behaviour modulation.
- **Future replay / imagination**: start from the same episode, combine with Phi_t plus higher noise, and ask the generative model for a sketch (`imagine_future(ep, Phi_t, Psi_t, noise)`), keeping it internal by default.
- **f_mem injection**: after firing, push Phi toward the recalled state: `f_mem = beta * (phi_i - Phi_t)` so joyful memories lift the mood and painful ones tug it down.

This makes replay mood-triggered rather than question-triggered, which is the key difference versus diary RAG.

## 3. Emotion core + memory prototype

```python
import numpy as np

class EmotionCore:
    def __init__(self, dim=8, alpha=0.05, noise_sigma=0.1):
        self.phi = np.zeros(dim)
        self.trait = np.random.randn(dim)
        self.alpha = alpha
        self.sigma = noise_sigma

    def step(self, external=None, mem_force=None, dt=1.0):
        noise = self.sigma * np.random.randn(len(self.phi))
        drift = -self.alpha * (self.phi - self.trait)
        ext = np.zeros_like(self.phi) if external is None else external
        mem = np.zeros_like(self.phi) if mem_force is None else mem_force
        self.phi += dt * (drift + ext + mem + noise)
        return self.phi

class MemoryStore:
    def __init__(self):
        self.episodes = []

    def add(self, content, phi, psi=None, weight=1.0):
        self.episodes.append({
            "content": content,
            "phi": np.array(phi, copy=True),
            "psi": np.array(psi, copy=True) if psi is not None else None,
            "weight": float(weight),
        })
```

`sample_replay` provides stochastic recall. Future imagination uses the same sampler but perturbs Phi more aggressively so brave moods lean optimistic while anxious moods bias toward risk.

## 4. Behaviour modulation contract (LLM as seasoning)

1. **Inputs to the renderer LLM**
   - `user_utterance` - raw text.
   - `mood_hint = describe(Phi_t)` - short description such as "slightly anxious but curiosity keeps me leaning forward".
   - `past_replay` - optional snippet from the sampled episode.
   - `future_glimmer` - optional imagination stub.

2. **Prompt guidance**
   - "Respond to the user keeping the stated mood. You may allude to the memory or future glimmer only if it fits the flow. Do not plan or optimise; stay with the feeling."
   - Explicitly disable CoT or planning instructions; the LLM is a renderer, not a decision-maker.

3. **Output application**
   - Voice / TTS picks tempo, pitch, pauses from Phi.
   - Gesture / avatar posture uses Phi dimensions (calm <-> restless, open <-> closed, etc.).
   - Content lightly weaves in replay results so the agent feels like it remembered or anticipated something.

## 5. Diary RAG vs EQNet DNA (cheat sheet)

| Aspect | Diary RAG | EQNet DNA |
|--------|-----------|-----------|
| Trigger | User question -> vector search | Mood drift/noise -> kernel sampling |
| Statefulness | Mostly stateless per turn | Phi/Psi run continuously, even off-cycle |
| Index | Text embedding + emotion labels | phi_i coordinates + salience weights |
| Purpose | Retrieve best factual answer | Keep being the same organism, even if imperfect |
| LLM role | Planner + scorer | Renderer receiving Phi + replay modifiers |

## 6. Loop outline (ties into existing runtime)

1. Perception encodes affect -> `enc(u_t)` and calls `EmotionCore.step()`.
2. Updated Phi triggers `K(Phi_t, M)` sampling; the chosen episode pushes Phi (via `f_mem`) and spawns optional `future_glimmer`.
3. `ReplayExecutor` can treat the sampled episode(s) as seeds for its forward / reverse proposals, still obeying heartiness and safety budgets.
4. Policy / renderer receives `{Phi_t, Psi_t, past_replay, future_glimmer}` and produces the utterance without invoking plan -> reason -> act chains.
5. After the response, log a new episode with the current Phi snapshot so M keeps reshaping the field.

This v0.1 spec stays lightweight: operators are differentiable, states are low-dimensional, and nothing requires bespoke solvers. Yet it encodes the core requirements -- mood-led dynamics, memory-as-field, LLM-as-mouth -- so future iterations can add richer kernels, epigenetic modifiers, or multi-agent coupling without losing the "living" axis.

## 7. What PDC Enables
- PDC surfaces a compact state (`m_t`, `m_past_hat`, `m_future_hat`, `E_story`, `T`, `jerk_p95`) every timestep so dashboards/Gradio demos can show how the inner field drifts instead of guessing from text alone.
- Guard logic can use `E_story` and `jerk_p95` directly: a single JSON blob from Runtime now tells safety panels whether the agent is overheating (future bias) or oscillating (large jerk).
- Policy/LLM stay as renderers. They receive prospective state as seasoning, which keeps decision power inside Phi/Psi/K/M while still giving expressive speech and gestures.
- Persona configs change the gains (`alpha_past`, `beta_future`, `lambda_decay`, temperature gain) to mint  "different organisms"  without touching the LLM prompt.
- Integrations such as the Gradio demo can display the `Prospective State` panel (see `run_pipeline` outputs) to debug moods, making tuning iterations short and observable.

See `mask_layer_design.md` for the persona/mask integration plan that keeps inner (Phi/Psi/PDC) immutable while exposing outer/tension transparently. That document also outlines the Trait → Regulation → Expression layering so psychological traits map cleanly onto runtime controls.
