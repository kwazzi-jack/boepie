---
type: Cab Note
title: casa.flagman
description: Why casa.flagman checkpoints are critical for reversible flagging workflows.
tags: [cab-note, casa, flagging, rollback]
managed_by: boepie
---

# casa.flagman

`casa.flagman` manages flag checkpoints so iterative flagging remains reversible. It is the backbone of safe experimentation in flagging-heavy workflows.

Use it over irreversible flagging when data quality strategy is still being tuned. Pair every major flagging stage with an explicit save point.

Gotchas observed in practice:

- Missing checkpoints force expensive reprocessing from raw data after a bad flag pass.
- Poor checkpoint naming makes recovery ambiguous in multi-branch reductions.
- Keep checkpoint intent documented in the recipe log.

# Citations

- smirnovRevisitingRadioInterferometer2011

