---
type: Cab Note
title: casa.split
description: When casa.split is used to isolate calibrated target subsets for downstream imaging.
tags: [cab-note, casa, split, data-management]
managed: boepie
---

# casa.split

`casa.split` creates focused subsets for downstream imaging and analysis. It is commonly used after transfer calibration to isolate target-ready data products.

Use it over keeping a monolithic dataset when you need cleaner operational boundaries and faster downstream iteration. Keep full data branches available for rollback or reprocessing.

Gotchas observed in practice:

- Split strategy can remove context needed for later troubleshooting.
- Multiple split branches require strict naming and provenance discipline.
- Validate subset integrity before long imaging runs.

# Citations

- smirnovRevisitingRadioInterferometer2011

