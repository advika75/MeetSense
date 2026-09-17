import subprocess
import sys


STEPS = [
    ("Audio preprocessing", "src/audio_preprocessing.py"),
    ("Voice activity detection", "src/vad.py"),
    ("Whisper transcription", "src/transcription.py"),
    ("Speaker diarization", "src/diarization.py"),
    ("Transcript-speaker alignment", "src/merge_transcript.py"),
    ("NLP preprocessing", "src/nlp_preprocessing.py"),
    ("TF-IDF keywords", "src/tfidf_keywords.py"),
    ("Topic modeling", "src/topic_modeling.py"),
    ("Action-item extraction", "src/action_item_extraction.py"),
    ("Transformer summarization", "src/summarization.py"),
]


print("=" * 70)
print("SPEAKER-AWARE MEETING ANALYSIS PIPELINE")
print("=" * 70)

print()

for step_number, (name, script) in enumerate(STEPS, start=1):

    print()
    print("=" * 70)
    print(f"STEP {step_number}/{len(STEPS)}: {name}")
    print("=" * 70)
    print()

    result = subprocess.run(
        [sys.executable, script],
        check=False
    )

    if result.returncode != 0:

        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)

        print()
        print(f"Failed step: {name}")
        print(f"Script: {script}")
        print(f"Exit code: {result.returncode}")

        sys.exit(result.returncode)

    print()
    print(f"✓ {name} completed successfully.")


print()
print("=" * 70)
print("COMPLETE PIPELINE FINISHED SUCCESSFULLY")
print("=" * 70)

print()
print("Generated outputs include:")

print("  • Preprocessed audio")
print("  • VAD speech segments")
print("  • Whisper transcript")
print("  • Speaker diarization")
print("  • Speaker-aware transcript")
print("  • NLP processed transcript")
print("  • TF-IDF keywords")
print("  • LDA topics")
print("  • Action items and decisions")
print("  • Transformer summary")

print()