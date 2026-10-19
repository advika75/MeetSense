# Live latency baseline

- Audio: `data/audio/meeting.wav`; simulated cap: 600s.
- Live UI chunk settings: 20s chunks, 3s overlap, 17s step; default three-chunk processing queue.
- Processed chunks: 17; captured audio: 600.0s; processed endpoint: 530.0s; queue overflow drops: 16.

- Expected capture windows: 36; not processed: 19 (16 queue-overflow drops and 3 still queued when Stop exited, derived from capture and processing counts).

| Stage | Mean ms | Median ms | P90 ms |
|---|---:|---:|---:|
| VAD | 122.4 | 108.9 | 163.8 |
| ASR | 2776.4 | 2113.4 | 4931.2 |
| Diarization | 35420.3 | 47.3 | 113804.5 |
| Alignment + NLP | 719.5 | 570.2 | 1091.2 |
| Total | 42088.0 | 12544.7 | 119871.8 |
| RTF | 2.104 | 0.627 | 5.994 |

- Chunks with RTF < 1.0: 10/17 (58.8%).
- Maximum end-to-end lag (chunk end to result): 287.50s.
- Stop-to-report build time, including live-model unload: 81.04s.
- Peak process RSS: 942.9 MB.
- Report extractor: `llm`.
- `report.json` validation: passed (`outputs/reports/live_latency_baseline_20261019/report.json`).
- Although 10/17 completed chunks had RTF < 1.0, processing did not sustain capture: 19 expected windows were not processed and max lag reached 287.50s. This is not a real-time result.
