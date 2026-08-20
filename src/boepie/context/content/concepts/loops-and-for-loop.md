---
type: Concept
title: Loops and for_loop
description: How for_loop expands recipe execution and what fails when loop variables or loop sources are invalid.
tags: [stimela, loops, for-loop, scatter-gather]
managed_by: boepie
---

# Loops and for_loop

A recipe-level `for_loop` repeats the same step graph over a sequence, optionally with scatter-style parallel workers. This is the core pattern for field-by-field or chunk-by-chunk processing without copy-pasting steps.

It matters because loops change substitution scope and output collection rules. You need loop-safe naming and deterministic output assembly, especially when later steps merge loop products.

```yaml
my_recipe:
  for_loop:
    var: item
    over: items
  inputs: {}
  steps:
    worker:
      cab: some.cab
      params: {}
```

If used wrongly, validation fails when the loop source is not defined as expected, or runtime fails when loop-scoped substitutions are referenced outside valid scope.

# Citations

- smirnovAfricanusIVStimela22025

