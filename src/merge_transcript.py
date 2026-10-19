import json
import os


# ============================================================
# FILE PATHS
# ============================================================

WHISPER_FILE = "outputs/transcripts/meeting.json"

DIARIZATION_FILE = "outputs/transcripts/diarization.json"

OUTPUT_FILE = "outputs/transcripts/speaker_transcript.json"


# ============================================================
# CALCULATE TEMPORAL OVERLAP
# ============================================================

def calculate_overlap(
    start1,
    end1,
    start2,
    end2
):
    """
    Calculate the temporal overlap between
    two time intervals.
    """

    overlap_start = max(
        start1,
        start2
    )

    overlap_end = min(
        end1,
        end2
    )

    return max(
        0.0,
        overlap_end - overlap_start
    )


# ============================================================
# ALIGN WHISPER SEGMENTS WITH SPEAKER SEGMENTS
# ============================================================

def align_segments(whisper_segments, diarization_segments):
    """Assign each Whisper segment the speaker label with maximum
    temporal overlap. Reused by both the batch pipeline (over the
    whole file) and the live pipeline (over the growing transcript,
    re-run against the latest diarization each time)."""

    speaker_transcript = []

    for whisper_segment in whisper_segments:

        whisper_start = float(
            whisper_segment["start"]
        )

        whisper_end = float(
            whisper_segment["end"]
        )

        text = str(
            whisper_segment.get(
                "text",
                ""
            )
        ).strip()

        if not text:
            continue

        best_speaker = "UNKNOWN"

        best_overlap = 0.0

        for diar_segment in diarization_segments:

            diar_start = float(
                diar_segment["start"]
            )

            diar_end = float(
                diar_segment["end"]
            )

            speaker = str(
                diar_segment["speaker"]
            )

            overlap = calculate_overlap(
                whisper_start,
                whisper_end,
                diar_start,
                diar_end
            )

            if overlap > best_overlap:

                best_overlap = overlap

                best_speaker = speaker

        speaker_transcript.append(
            {
                "start": round(
                    whisper_start,
                    3
                ),

                "end": round(
                    whisper_end,
                    3
                ),

                "speaker": best_speaker,

                "text": text,

                "speaker_overlap": round(
                    best_overlap,
                    3
                )
            }
        )

    return speaker_transcript


# ============================================================
# BATCH CLI ENTRY POINT (unchanged behavior)
# ============================================================

def main():

    print("=" * 60)
    print("SPEAKER-AWARE TRANSCRIPT ALIGNMENT")
    print("=" * 60)

    print()
    print("Loading Whisper transcript...")

    with open(
        WHISPER_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        whisper_data = json.load(f)

    if isinstance(whisper_data, list):

        whisper_segments = whisper_data

    elif isinstance(whisper_data, dict):

        whisper_segments = whisper_data["segments"]

    else:

        raise ValueError(
            "Unexpected Whisper JSON format."
        )

    print("Loading PyAnnote diarization...")

    with open(
        DIARIZATION_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        diarization_data = json.load(f)

    if isinstance(diarization_data, list):

        diarization_segments = diarization_data

    elif isinstance(diarization_data, dict):

        if "segments" in diarization_data:

            diarization_segments = (
                diarization_data["segments"]
            )

        else:

            raise ValueError(
                "Unexpected diarization JSON format."
            )

    else:

        raise ValueError(
            "Unexpected diarization JSON format."
        )

    print()
    print(
        "Whisper segments:",
        len(whisper_segments)
    )

    print(
        "Diarization segments:",
        len(diarization_segments)
    )

    print()
    print("Aligning Whisper segments with speakers...")

    speaker_transcript = align_segments(
        whisper_segments,
        diarization_segments
    )

    os.makedirs(
        "outputs/transcripts",
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            speaker_transcript,
            f,
            indent=2,
            ensure_ascii=False
        )

    speakers = sorted(
        set(
            segment["speaker"]
            for segment in speaker_transcript
        )
    )

    print()
    print("=" * 60)
    print("ALIGNMENT COMPLETE")
    print("=" * 60)

    print()
    print(
        "Speaker-aware segments:",
        len(speaker_transcript)
    )

    print(
        "Speakers detected:",
        len(speakers)
    )

    print()
    print("Speaker labels:")

    for speaker in speakers:

        print(
            "  -",
            speaker
        )

    print()
    print("First 10 aligned segments:")
    print("-" * 60)

    for segment in speaker_transcript[:10]:

        print(
            f"{segment['start']:8.2f}s - "
            f"{segment['end']:8.2f}s | "
            f"{segment['speaker']:12} | "
            f"{segment['text']}"
        )

    print()
    print("=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
