---
type: Cab Note
title: pfb.grid gridding stage
description: Role of pfb.grid in preparing data for optimization-based reconstruction.
tags: [cab-note, pfb-imaging, gridding, reconstruction]
managed_by: boepie
---

# pfb.grid

`pfb.grid` builds gridded intermediate products used by later optimization stages in a PFB chain. It is a core transition from visibility-domain inputs to reconstruction-domain work products.

Use it over standard imager gridding paths when running a full `pfb-imaging` strategy. For routine continuum products where turnaround dominates, `wsclean` is usually the faster path.

Gotchas observed in practice:

- Input consistency with initialization and later steps is essential; ad hoc invocation is fragile.
- Diagnostic checks should confirm gridding products before long iterative reconstruction.
- Version and environment alignment matter; use managed recipe execution.

# Citations

- besterAfricanusIIIPfbimagingA2026

