---
type: Cab Note
title: casa.bandpass
description: Why casa.bandpass is used before transfer across frequency and what breaks when omitted.
tags: [cab-note, casa, bandpass, calibration]
managed_by: boepie
---

# casa.bandpass

`casa.bandpass` solves spectral response behavior needed for reliable wide-frequency transfer. It is a standard step in cross-calibration chains before target correction.

Use it over skipping spectral calibration when broad-band target fidelity matters. In narrow and stable subsets it may be less dominant, but omission risks frequency-structured artefacts.

Gotchas observed in practice:

- Poor calibrator signal support produces unstable spectral transfer.
- Bandpass solve quality must be checked before gain transfer decisions.
- Revisit this step when scan selection or preprocessing changes.

