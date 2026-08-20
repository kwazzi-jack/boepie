---
type: Cab Note
title: pfb.restore image product stage
description: How pfb.restore produces interpretable image products from reconstruction outputs.
tags: [cab-note, pfb-imaging, restore, imaging]
managed_by: boepie
---

# pfb.restore

`pfb.restore` converts reconstruction products into final image-domain outputs suitable for downstream analysis. It is the handoff point from optimization internals to interpretable maps.

Use it over ad hoc post-processing when finishing a PFB chain and preparing comparable products across runs. Keep final image assembly inside the same managed workflow for reproducibility.

Gotchas observed in practice:

- Restored outputs are only as reliable as upstream calibration and reconstruction decisions.
- Product naming and staging should be explicit to avoid confusion with alternative imaging branches.
- Compare restored products against visibility diagnostics before catalog extraction.

# Citations

- besterAfricanusIIIPfbimagingA2026

