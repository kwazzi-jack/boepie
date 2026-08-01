---
type: Cab Note
title: pfb.init setup stage
description: What pfb.init prepares in a pfb-imaging chain and when to include it.
tags: [cab-note, pfb-imaging, setup, reconstruction]
managed: boepie
---

# pfb.init

`pfb.init` prepares the state and products expected by downstream `pfb-imaging` steps. It is the normal entry point for a PFB reconstruction chain.

Use it over direct entry into later `pfb` steps when starting from raw calibrated visibilities. Skip only if an existing compatible PFB working state is explicitly available and verified.

Gotchas observed in practice:

- Reusing stale initialization state across materially different data selections can invalidate downstream products.
- PFB chains are sensitive to workflow consistency; keep all major steps in one controlled recipe.
- Pull current control surface from live schema tools before tuning runs.

# Citations

- besterAfricanusIIIPfbimagingA2026

