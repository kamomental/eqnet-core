# -*- coding: utf-8 -*-
"""
Transcribe an audio file, optionally perform diarisation, and derive a persona
profile using the new speech preference adapter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Set

from emot_terrain_lab.audio.asr_pipeline import ASRConfig, ASRTranscriber
from emot_terrain_lab.persona.speech_adapter import SpeechPreferenceExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persona YAML from an audio file (ASR + optional diarisation)."
    )
    parser.add_argument("audio", type=Path, help="Path to the audio file (wav/mp3/m4a).")
    parser.add_argument("--model", default="medium", help="faster-whisper model size (default: medium).")
    parser.add_argument("--device", default="cpu", help="Device string for faster-whisper.")
    parser.add_argument("--compute-type", default="int8", help="Compute type (int8/int16/float16/float32).")
    parser.add_argument("--language", help="Optional language hint (ja, en, ja-JP, ...).")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding.")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="VAD threshold for faster-whisper.")
    parser.add_argument("--diarize", action="store_true", help="Enable diarisation via pyannote.audio.")
    parser.add_argument("--diarize-model", help="Custom diarisation model id.")
    parser.add_argument("--diarize-token", help="Auth token for diarisation model (if required).")
    parser.add_argument(
        "--speaker",
        action="append",
        dest="speakers",
        help="Target speaker(s); can be supplied multiple times. If omitted, all speakers are used.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum ASR confidence (0-1).")
    parser.add_argument("--persona-out", type=Path, help="Optional path to write the derived persona YAML.")
    parser.add_argument("--no-preview", action="store_true", help="Do not print the persona YAML to stdout.")
    parser.add_argument("--print-transcript", action="store_true", help="Print recognised segments to stderr.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ASRConfig(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        language=args.language,
        vad_threshold=args.vad_threshold,
        diarization=args.diarize,
        diarization_model=args.diarize_model,
        diarization_auth_token=args.diarize_token,
    )
    transcriber = ASRTranscriber(cfg)
    segments = transcriber.transcribe(args.audio)

    if args.print_transcript:
        for seg in segments:
            print(
                f"[{seg.start:.2f}-{seg.end:.2f}] {seg.speaker} (conf={seg.confidence:.2f}): {seg.text}",
                file=sys.stderr,
            )

    target_speakers: Optional[Set[str]] = set(args.speakers) if args.speakers else None
    extractor = SpeechPreferenceExtractor(
        target_speakers=target_speakers,
        min_confidence=args.min_confidence,
    )
    draft = extractor.to_persona(segments, lang_hint=args.language)

    if not args.no_preview:
        print(draft.to_yaml())
        if draft.notes:
            print("# notes:", file=sys.stdout)
            for note in draft.notes:
                print(f"# - {note}", file=sys.stdout)

    if args.persona_out:
        draft.save(args.persona_out)


if __name__ == "__main__":
    main()
