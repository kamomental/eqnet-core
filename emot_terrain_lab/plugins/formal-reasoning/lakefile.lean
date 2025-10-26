import Lake
open Lake DSL

package «eqnet_formal» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "stable"

@[default_target]
lean_lib EQNetSafety
