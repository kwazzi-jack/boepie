---
type: Playbook
title: Cross-calibration workflow
description: Transfer calibrator-derived solutions to targets and prepare corrected data for imaging.
tags: [playbook, cross-calibration, calibrators, transfer]
managed_by: boepie
---

# Cross-calibration workflow

Task: derive stable gains on calibrators and transfer them to target data before self-calibration and deep imaging.

Cab sequence:

1. `casa.listobs` - confirm scan layout and calibrator-target segmentation.
2. `casa.setjy` - set the calibrator flux model basis used by all downstream gain solves.
3. `casa.gaincal` - solve complex gains on calibrator scans.
4. `casa.bandpass` - solve spectral response terms needed for robust transfer across frequency.
5. `casa.fluxscale` - place gain scale on the desired flux reference using calibrator relationships.
6. `casa.applycal` - transfer calibrator solutions to target and calibrator datasets.
7. `casa.split` - write a calibrated target subset for imaging and selfcal workflows.
8. `casa.mstransform` - regrid or average into an analysis-ready target product when required.

Decision points:

- If calibrator residuals show unresolved structure after first solutions, refine calibrator modeling before transfer.
- If target fields are observed far from calibrator conditions in time or direction, expect transfer limits and plan early target selfcal.
- If post-transfer diagnostics show frequency-dependent artefacts, revisit spectral calibration before proceeding.

# Citations

- smirnovRevisitingRadioInterferometer2011
