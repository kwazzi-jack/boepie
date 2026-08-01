---
type: Cab Note
title: PyBDSF catalog extraction
description: How bdsf.catalog turns final images into source products and where extraction bias appears.
tags: [cab-note, bdsf, catalogs, source-finding]
managed: boepie
---

# bdsf.catalog

`bdsf.catalog` runs PyBDSF source finding and catalog writing on restored images. It is typically the handoff step from imaging to science catalog analysis.

Use it over custom source-picking scripts when you need repeatable detection and export behavior. Use custom analysis only after baseline extraction, for project-specific post-processing.

Gotchas observed in practice:

- Extraction quality depends strongly on residual artefacts and local noise variation from upstream imaging.
- Catalog interpretation should include image-domain quality checks, not threshold logic alone.
- Keep detection controls synchronized with current package behavior via live schema queries.

# Citations

- hogbomApertureSynthesisNonRegular1974

