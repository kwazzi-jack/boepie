---
type: Playbook
title: Continuum imaging workflow
description: End-to-end strategy for calibrated continuum images with branch points for difficult fields.
tags: [playbook, continuum, imaging, calibration]
managed_by: boepie
---

# Continuum imaging workflow

Task: produce a science-ready continuum image from calibrated visibilities, with masks and source products for follow-up analysis.

Cab sequence:

1. `msutils.summary` - establish scan, field, and data sanity before any destructive action.
2. `tricolour` - remove obvious radio-frequency interference so calibration and imaging solve cleaner residual structure.
3. `quartical` - solve and apply direction-independent gains using the current sky model.
4. `wsclean` - form deep deconvolved continuum images that drive all later quality checks.
5. `breizorro` - build imaging masks from current products to stabilize deeper deconvolution passes.
6. `wsclean` - re-image with mask support to suppress artefacts and recover fainter structure.
7. `bdsf.catalog` - extract catalog-ready source products from the final restored image.
8. `shadems` - produce fast visibility diagnostics to confirm no strong residual calibration pathologies remain.

Decision points:

- If residual images show broad diffuse structure that is poorly represented by compact components, branch to a `pfb-imaging` chain for reconstruction designed around large-scale emission.
- If bright off-axis sources dominate sidelobes after the first deep image, insert additional direction-dependent calibration before final imaging.
- If aggressive flagging removes large contiguous chunks in time or frequency, re-check calibration stability before committing to final deconvolution.

# Citations

- offringaWscleanImplementationFast2014
- kenyonAfricanusIIQuartiCal2025
- besterAfricanusIIIPfbimagingA2026
