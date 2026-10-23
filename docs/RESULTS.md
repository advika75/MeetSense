# MeetSense results index

All results come from one meeting recording. Live mode was tested by file
replay, not a real microphone. Live DER values are proxies against offline
PyAnnote output.

## Takeaway extraction (rule-based vs LLM vs LLM+verify)
- Summary and per-item false positives/misses: [outputs/eval/takeaways_eval.md](../outputs/eval/takeaways_eval.md)
- Labels: [data/labels/meeting1.json](../data/labels/meeting1.json)
- Reports: [meeting_001](../outputs/reports/meeting_001/), [meeting_001_llm](../outputs/reports/meeting_001_llm/), [meeting_001_llm_verify](../outputs/reports/meeting_001_llm_verify/)

## Live mode latency
- Baseline: [outputs/live/baseline_report.md](../outputs/live/baseline_report.md), [report.json](../outputs/reports/live_latency_baseline_20261019/report.json)
- Async PyAnnote: [outputs/live/after_report.md](../outputs/live/after_report.md), [report.json](../outputs/reports/live_latency_after_20261020/report.json)
- Online embeddings: [outputs/live/online_report.md](../outputs/live/online_report.md), [report.json](../outputs/reports/live_online_embedding_20261020/report.json)

## Offline pipeline (WER, DER, ROUGE, action items)
- [outputs/evaluations/](../outputs/evaluations/); see "Actual Results" in [README.md](../README.md)
