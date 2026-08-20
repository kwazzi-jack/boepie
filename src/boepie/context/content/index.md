# Stimela knowledge base

Selective map of the curated knowledge bundle. Read this file first, then
jump straight to the file that matches your question - do not browse the
whole tree.

## Escalation ladder

| Question shape | Where to look |
|-----------------|----------------|
| "What are this cab's parameters / defaults / choices?" | `get_cab_schema`, `get_cab_docs`, `get_cab_params` (live, authoritative - never this bundle) |
| "How do I structure this pipeline / task?" | `playbooks/` |
| "What does this recipe-language feature mean, how do I write it?" | `concepts/` |
| "What does this specific cab do, when do I reach for it, what are its gotchas?" | `cabs/` |
| "How do I use stimela / quartical / wsclean itself?" | `search_docs`, then `read_docs` |
| "Why does this algorithm behave this way?" | `literature/` stub, then `read_literature` |
| "What did this user write down themselves?" | `search_notes`, then `read_notes` |
| "What is even in the corpus?" | `list_corpus` |

## Layout

- `concepts/` - recipe language, substitution, backends, loops
- `playbooks/` - task-to-cab strategies (imaging, selfcal, flagging)
- `cabs/` - what/when/gotchas per cab; no parameter tables, cites papers
- `literature/` - one stub per paper: abstract + `read_literature` handle
- `apply-log.md` - update history, newest first
- `manifest.json` - bundle_version, cultcargo_version, boepie_version, generated_at

## Read handles

Corpus documents (literature, docs, notes) are addressed by an opaque `id`
such as `aB3dE9fGhI`, which is stable across renames and regrouping. Copy it
from a `search_*` hit's `read:` line, or from `list_corpus`. A citekey,
arXiv id, or exact title also resolves, which is why the stubs in
`literature/` can cite a citekey directly - but never invent a handle.

Note that this bundle is different: `search_context` returns file paths, and
you open those with your own file tools. There is no `read_context`.

## Rule

Parameter values, defaults, and choices are never stored here: they drift
with the installed cult-cargo version and go stale silently. Always get them
live through the MCP cab tools (L1), and treat this bundle (L2) as strategy
and concepts only.
