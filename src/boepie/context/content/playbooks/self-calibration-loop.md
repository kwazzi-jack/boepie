---
type: Playbook
title: Self-calibration loop
description: Iterative model-calibrate-image loop for improving dynamic range on target fields.
tags: [playbook, selfcal, calibration, imaging]
managed_by: boepie
---

# Self-calibration loop

Task: iteratively improve model and gains until residual structure is noise-like and image artefacts stop improving materially.

Cab sequence:

1. `msutils.copycol` - preserve an input visibility state so each loop can be compared and safely rolled back.
2. `wsclean` - generate the current model and residual reference from the working data state.
3. `quartical` - solve gains against the current model and update corrected data.
4. `wsclean` - re-image with updated calibration to assess dynamic range gains.
5. `taql.update` - apply targeted visibility edits only when diagnostics indicate a narrow, correctable defect.
6. `shadems` - inspect per-baseline and per-time behavior before deciding on another loop.

Decision points:

- If residuals remain dominated by coherent sidelobe patterns near bright sources, continue looping.
- If residuals become noise-like and image changes are marginal between iterations, stop to avoid overfitting.
- If solutions become unstable after a loop while data quality metrics do not improve, revert and shorten loop aggressiveness.

# Citations

- salviniFastGainCalibration2014
- smirnovRadioInterferometricGain2015
- kenyonAfricanusIIQuartiCal2025

