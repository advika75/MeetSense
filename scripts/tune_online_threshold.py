"""Tune live online-speaker cosine threshold on the first 180s."""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from diarization import load_diarization_pipeline, embed_speech_segments
from vad import load_silero_vad, detect_speech


def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def optimal_accuracy(predicted, reference):
    p = sorted(set(predicted))
    r = sorted(set(reference))
    matrix = np.zeros((len(p), len(r)), dtype=np.int64)
    pi = {x: i for i, x in enumerate(p)}
    ri = {x: i for i, x in enumerate(r)}
    for x, y in zip(predicted, reference):
        matrix[pi[x], ri[y]] += 1
    rows, cols = linear_sum_assignment(-matrix)
    return int(matrix[rows, cols].sum()), len(predicted)


def main():
    audio, sr = sf.read(os.path.join(ROOT, "data/audio/meeting.wav"), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio[:180 * sr]
    offline = json.load(open(os.path.join(ROOT, "outputs/transcripts/diarization.json")))
    vad_model = load_silero_vad()
    vad = detect_speech(audio, sr, model=vad_model)
    segments = []
    for item in vad:
        start = item["start"] / sr
        end = item["end"] / sr
        if end - start < 1.0:
            continue
        row = {"start": start, "end": end}
        matches = [(overlap(row, ref), ref["speaker"]) for ref in offline if overlap(row, ref) > 0]
        if matches:
            row["reference"] = max(matches)[1]
            segments.append(row)
    pipeline = load_diarization_pipeline()
    vectors = embed_speech_segments(pipeline, audio, [(s["start"], s["end"]) for s in segments], sr)
    del pipeline

    results = []
    for threshold in np.arange(0.05, 0.901, 0.025):
        centroids = []
        labels = []
        for vector in vectors:
            if not centroids:
                cluster = 0
            else:
                sims = [float(np.dot(vector, center)) for center in centroids]
                nearest = int(np.argmax(sims))
                if sims[nearest] >= threshold or len(centroids) >= 8:
                    cluster = nearest
                else:
                    cluster = len(centroids)
            if cluster == len(centroids):
                centroids.append(vector.copy())
            else:
                center = 0.9 * centroids[cluster] + 0.1 * vector
                centroids[cluster] = center / max(float(np.linalg.norm(center)), 1e-12)
            labels.append(f"S{cluster}")
        matched, total = optimal_accuracy(labels, [s["reference"] for s in segments])
        results.append((matched / total if total else 0.0, -abs(len(set(labels)) - len(set(s["reference"] for s in segments))), -abs(threshold - 0.35), -threshold, threshold, len(set(labels)), matched, total))
    best = max(results)
    print(f"TUNED_THRESHOLD={best[4]:.3f}")
    print(f"SEGMENT_AGREEMENT={best[6]}/{best[7]} ({best[0]:.1%})")
    print(f"ONLINE_SPEAKERS={best[5]}")
    print("GRID=" + json.dumps([{"threshold": round(x[4], 3), "agreement": x[0], "speakers": x[5]} for x in sorted(results, key=lambda x: x[4])]))


if __name__ == "__main__":
    main()
