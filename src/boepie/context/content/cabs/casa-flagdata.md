---
type: Cab Note
title: casa.flagdata
description: Where casa.flagdata complements automated flaggers for targeted manual cleanup.
tags: [cab-note, casa, flagdata, rfi]
managed_by: boepie
---

# casa.flagdata

`casa.flagdata` performs targeted rule-based or manual flagging and complements automated first-pass tools. It is best used after broad automated rejection to handle known residual defects.

Use it over `tricolour` when you need explicit hand-tuned operations tied to observed data features. Use `tricolour` first for bulk statistical cleanup.

Gotchas observed in practice:

- Broad manual criteria can remove scientifically useful data without obvious immediate warning.
- Complex sequential operations are hard to audit unless captured in recipe form.
- Always compare pre and post flagging diagnostics before committing.

# Citations

- mitchellRealTimeCalibrationMurchison2008

