# Live latency after

- Audio: `data/audio/meeting.wav`; simulated cap: 600s.
- Live UI chunk settings: 20s chunks, 3s overlap, 17s step; three-item queue with backlog merging; asynchronous diarization; 90s buffer cap.
- Processed batches: 15 (36 source windows); captured audio: 600.0s; processed endpoint: 600.0s; dropped chunks: 0; merged windows: 21.
- Expected capture windows: 36; not processed: 0 (queue overflow: 0; queued when Stop exited: 0).

| Stage | Mean ms | Median ms | P90 ms |
|---|---:|---:|---:|
| VAD | 479.4 | 170.2 | 1254.6 |
| ASR | 6490.1 | 4020.2 | 13328.8 |
| Diarization | 102.6 | 38.5 | 290.9 |
| Alignment + NLP | 1273.3 | 778.8 | 2619.8 |
| Total | 72277.7 | 25014.6 | 187898.9 |
| RTF | 3.020 | 0.887 | 8.199 |

- Chunks with RTF < 1.0: 8/15 (53.3%).
- Maximum end-to-end lag (chunk end to result): 543.10s.
- Stop-to-report time: 581.00s; diarization stop timeout: no; provisional labels in report: no.
- Mean time until speaker labels are refined: 128.36s (since diarization request).
- Peak process RSS: 945.7 MB.
- `report.json` validation: passed (`outputs/reports/live_latency_after_20261020/report.json`).
- Report extractor: `llm`.
- RTF < 1.0 for most chunks: yes (8/15 chunks).

| Metric | Before | After |
|---|---:|---:|
| Windows processed / dropped / merged / queued | 17 / 16 / 0 / 3 | 36 / 0 / 21 / 0 |
| RTF mean / median / p90 | 2.104 / 0.627 / 5.994 | 3.020 / 0.887 / 8.199 |
| Max end-to-end transcript lag | 287.50s | 543.10s |
| Mean time until speaker labels refined | unavailable | 128.36s |
| Stop-to-report time | 81.04s | 581.00s |
| Peak RSS | 942.9 MB | 945.7 MB |

- Final speaker-label agreement with baseline: 32/36 comparable segments (88.9% same speaker).
- Real-time claim: no (zero drops and max transcript lag 543.10s).
