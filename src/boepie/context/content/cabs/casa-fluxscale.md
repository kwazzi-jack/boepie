---
type: Cab Note
title: casa.fluxscale
description: How casa.fluxscale propagates calibrator flux scale and where transfer assumptions fail.
tags: [cab-note, casa, fluxscale, transfer]
managed_by: boepie
---

# casa.fluxscale

`casa.fluxscale` propagates an established calibrator flux scale through calibration products for transfer. It is a bridge between raw gain solutions and physically interpretable target scaling.

Use it over manual scale assumptions whenever absolute or cross-epoch comparability matters. Skip only in workflows where relative calibration is explicitly acceptable.

Gotchas observed in practice:

- Transfer quality degrades when calibrator relationships are weak or inconsistent.
- Downstream imaging residuals can reveal flux transfer faults not obvious at solve time.
- Record flux-scale assumptions with final products.

