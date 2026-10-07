# Use of AI tools

This assignment permits LLMs and coding assistants provided the assistance is cited.
This file records how Claude Code (Anthropic, model Claude Fable 5.1) was used, so
that the grader can see exactly which parts were tool-assisted.

| Area | Tool involvement | Author involvement |
|---|---|---|
| Experiment design | Proposed the run matrix (Part I controls, the 3 × 2 embedding factorial, Part III representation set), the leakage checklist and the evaluation protocols after the author fixed the datasets, compute, architecture for Part II and reporting format | Chose datasets, compute budget, LSTM for Part II, report toolchain; approved and adjusted the plan |
| Code (`lm/`, `scripts/`, `tests/`) | Drafted all modules and the unit tests | Ran every experiment, inspected outputs, checked results for sanity |
| Result tables and figures | `scripts/make_report_assets.py` generates them mechanically from `results/` | Verified numbers against the raw JSON/CSV files |
| Report text | Drafted from the generated tables; the observations-vs-hypotheses structure was set by the author | Edited the interpretation; the conclusions reflect the author's understanding |

No AI tool was used to fabricate or alter any result; every number in the report can
be regenerated with `run_all.sh` (to within about one perplexity point, since MPS training is
not bit-reproducible).
