---
type: Cab Note
title: pfb.kclean optimization stage
description: When pfb.kclean helps recover complex structure and where iteration can become unstable.
tags: [cab-note, pfb-imaging, optimization, deconvolution]
managed_by: boepie
---

# pfb.kclean

`pfb.kclean` performs an iterative reconstruction stage in the `pfb-imaging` pipeline. It is used when sparse or structured optimization is preferred over classic clean-family behavior.

Use it over repeated `wsclean` refinement in fields where diffuse structure or algorithmic priors are central to science goals. Use `wsclean` when robust fast production images are the priority.

Gotchas observed in practice:

- Iterative optimization can drift if upstream calibration is inconsistent.
- Monitoring convergence behavior is essential; long runs are not automatically better runs.
- Keep reconstruction diagnostics and checkpoints at each major stage.

# Citations

- besterAfricanusIIIPfbimagingA2026

