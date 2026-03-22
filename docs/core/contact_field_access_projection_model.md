# Contact Field And Access Projection Model

Date: 2026-03-18

## Why This Layer Exists

`qualia membrane` should not jump directly from a total internal state into a
foreground workspace.

The engineering split is easier to reason about if there is an explicit
pre-workspace layer:

- `Contact Field`
  - local contact-point emergence
- `Access Projection`
  - shaping those points into access-ready regions

This keeps the system closer to:

- local triggering
- structured integration
- selective foreground entry

rather than a single opaque access score.

## Stack Position

```text
terrain / body / relation / memory / scene
    -> contact field
    -> contact dynamics
    -> access projection
    -> access dynamics
    -> conscious workspace
    -> interaction option search
    -> policy packet
    -> articulation shell
```

## Contact Field

`Contact Field` is a small set of local contact points.

Each point is a compact engineering representation of:

- where something touched the current state
- how strongly it touched
- how much defensive or reportability pressure came with it

Current `ContactPoint` shape:

- `point_id`
- `label`
- `source_modality`
- `intensity`
- `temporal_kernel_response`
- `local_terrain_gradient`
- `local_curvature`
- `relation_tag`
- `scene_tag`
- `uncertainty`
- `ambiguity`
- `defensive_salience`
- `binding_tags`

This is intentionally still local and pre-workspace.

## Access Projection

`Access Projection` takes contact points and turns them into access-ready
regions.

In the current implementation it can also read `contact dynamics`, so access is
not limited to one-shot point intensity. Re-entry and carryover can slightly
change which regions remain actionable.

It does not yet claim full conscious foreground.

It decides, in a bounded engineering sense:

- what is reportable
- what should stay withheld
- what is still actionable even if not reportable

Current `AccessRegion` shape:

- `region_id`
- `label`
- `source`
- `activation`
- `reportable`
- `withheld`
- `actionable`
- `binding_tags`

Current `AccessProjection` shape:

- `projection_mode`
- `dominant_region`
- `reportable_slice`
- `withheld_slice`
- `actionable_slice`
- `regions`

## Access Dynamics

`Access Dynamics` is the membrane-inertia layer above `Access Projection`.

It does not replace projection. It keeps the projection from collapsing into a
one-shot gate by adding:

- membrane inertia
- gating hysteresis
- protective filtering
- stabilized access regions

Short form:

- `Access Projection`
  - immediate access shaping
- `Access Dynamics`
  - short-horizon membrane carryover

Current `AccessDynamicsState` shape:

- `dynamics_mode`
- `membrane_inertia`
- `gating_hysteresis`
- `protective_filter`
- `stabilized_regions`
- `reportable_slice`
- `withheld_slice`
- `actionable_slice`

## Engineering Meaning

This split is useful because it prevents one collapse:

`something became active -> therefore it must be speakable`

Instead the chain becomes:

`contact -> access -> workspace -> policy`

which makes room for:

- withheld but active content
- guarded but still actionable content
- protective pressure before explicit report

## Relation To Conscious Workspace

`Conscious Workspace` now consumes `AccessProjection` rather than inventing its
foreground entirely from downstream policy pressure.

Short form:

- `Contact Field`
  - local emergence
- `Contact Dynamics`
  - temporal carryover and re-entry
- `Access Projection`
  - access shaping
- `Access Dynamics`
  - membrane inertia and gating hysteresis
- `Conscious Workspace`
  - held foreground

## Current Scope

This is still a `v0` engineering layer.

It does provide:

- explicit local contact points
- explicit access-ready regions
- explicit membrane inertia over access-ready regions
- guarded/reportable/actionable splitting before workspace

It does not yet provide:

- full contact-point dynamics over time
- continuous membrane deformation equations
- learned contact topology
- nightly reconsolidation of access regions

That is acceptable for now.
The purpose of this layer is to make the route from terrain to workspace
explicit and testable.
