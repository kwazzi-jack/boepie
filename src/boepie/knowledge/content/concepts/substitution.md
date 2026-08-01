---
type: Concept
title: Substitution and evaluation
description: How recipe and step substitutions are resolved, and why unresolved values fail at runtime.
tags: [stimela, substitution, evaluation, runtime]
managed: boepie
---

# Substitution and evaluation

Substitution lets a recipe derive step values from recipe state and prior step state. In practice, this is how data products move through a pipeline without hard-coded paths in every step.

It matters because substitution is evaluated in context. A value that exists at recipe scope may not exist at the current step scope, and unresolved values do not silently produce good science output.

```yaml
imaging_recipe:
  assign:
    output_dir: products
  steps:
    make_image:
      cab: some.imager
      params:
        output: =recipe.output_dir + "/image"
```

If used wrongly, the run fails with unresolved substitution or assignment errors. The usual root cause is scope mismatch, missing assignment, or a value expected from a step that never ran.

# Citations

- smirnovAfricanusIVStimela22025

