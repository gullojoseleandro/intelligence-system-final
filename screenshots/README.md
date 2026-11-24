# Dashboard evidence pending

This directory intentionally contains no fabricated screenshots. Add genuine LangSmith captures after a five-request run in `MODEL_MODE=openai` with tracing enabled:

1. A trace tree showing supervisor, research, analysis, model call, tokens, and per-execution cost.
2. A tracing-project dashboard showing the five runs and cost metrics.
3. The five-run p95 latency evidence. LangSmith currently exposes p50/p99 (not p95) in its custom latency chart; pair a dashboard trace-duration capture with `load_test.py`'s clearly labeled nearest-rank p95 output, or use a platform that exposes p95 directly.

Do not submit offline-demo traces as proof of real model costs.
