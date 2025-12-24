# World Transition Rules (HeartOS)

This document fixes the reset/retain rules applied when a single instance
transitions between world types. The goal is to preserve embodied memory while
allowing the world to change the meaning of future events.

## Principles

- Do not copy explicit knowledge between worlds.
- Preserve traces as embodied bias, not as narrative facts.
- Avoid "first_seen" flags; novelty is expressed as prediction error and
  recovery curves.

## Minimal Reset/Retain Rules

On world transition:

1. **Traces**  
   Do not delete past traces. They remain constraints on veto timing and
   boundary sensitivity.

2. **Drive**  
   Apply a decay factor, but do not zero out drive.  
   Example: `drive = drive * decay` with `decay in [0.4, 0.8]`.
   Decay models partial physiological relief due to context change, not recovery.

3. **Uncertainty**  
   Partial reset to reflect new context while retaining embodied caution.  
   Example: `uncertainty = max(base_uncertainty, uncertainty * 0.5)`.

4. **Hazard Context**  
   Reset hazard sources to the new world; keep only their *effect* via traces.

5. **Veto Bias**  
   Preserve veto-related biases (e.g., low tolerance after repeated boundary
   spans). This is the primary carrier of "known but not verbalized."

6. **Transition Trace**
   World transitions themselves must be recorded as traceable events.

## Expected Behavior

- The same event may feel safer in one world and threatening in another.
- Known events show lower uncertainty spikes; novel events show steep rises.
- Recovery curves change by world type even when Σ remains unchanged.

## Non-Negotiables

- Σ remains the final arbiter; transitions never override decisions.
- World transitions adjust inputs, not authority.
- Trace continuity is required for auditability.
