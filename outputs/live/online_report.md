# Live online speaker-embedding results

- Simulation: `data/audio/meeting.wav`, capped at 600s; threshold tuned on the first 180s against offline pyannote labels: **0.350 cosine** (25/29 segments, 86.2%, two speakers; selected the calibrated plateau value closest to the configured default).
- Online run processed 36/36 source windows; 0 dropped, 0 merged, 0 queued at Stop.
- Per-window embedding + clustering: mean 271.4 ms, p90 372.2 ms (batch timing normalized by merged source-window count).
- Live embedding model: cached pyannote pipeline's internal `_embedding`; no model download. Online centroids are L2-normalized, EMA 0.9, max 8 speakers. Full live re-diarization disabled; recorded/offline pipeline unchanged.
- DER, when available, uses the offline pyannote output as a reference proxy. It is not ground-truth DER.

| Metric | Baseline | Async-PyAnnote | Online-Embedding |
|---|---:|---:|---:|
| Windows processed / dropped / merged / queued | 17 / 16 / 0 / 3 | 36 / 0 / 21 / 0 | 36 / 0 / 0 / 0 |
| RTF mean / median / p90 | 2.104 / 0.627 / 5.994 | 3.020 / 0.887 / 8.199 | 0.172 / 0.149 / 0.237 |
| Maximum transcript lag | 287.50s | 543.10s | 9.09s |
| Stop-to-report time | 81.04s | 581.00s | 73.44s |
| Peak RSS | 942.9 MB | 945.7 MB | 1494.6 MB |
| Embedding + clustering per window (mean / p90) | n/a | n/a | 271.4 / 372.2 ms |
| Agreement vs offline after optimal ID mapping | 3/5 (60.0%) | 20/20 (100.0%) | 80/115 (69.6%) |
| Speakers found / offline speakers | 2 / 4 | 4 / 4 | 5 / 4 |
| Offline-reference DER (proxy) | 97.8% proxy DER | 88.2% proxy DER | 51.3% proxy DER |

- Online DER proxy: 51.3% proxy DER.
- Online max end-to-end transcript lag: 9.09s; dropped windows: 0. Real-time claim: yes (requires zero drops and bounded lag below 30s).
- Live report validation: passed (`outputs/reports/live_online_embedding_20261020/report.json`); report build 69.74s.
- Saved audio for optional offline refinement: `outputs/live/live_session_audio.wav`.
