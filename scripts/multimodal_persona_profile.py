# -*- coding: utf-8 -*-
"""
Generate persona profile from multiple modalities (text/audio/images).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from emot_terrain_lab.audio.asr_pipeline import ASRConfig, ASRTranscriber
from emot_terrain_lab.persona.multimodal_adapter import MultimodalPreferenceBuilder
from emot_terrain_lab.persona.speech_adapter import SpeechPreferenceExtractor
from emot_terrain_lab.vision.caption_pipeline import CaptionConfig, ImageCaptioner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate persona YAML from text/audio/images.")
    parser.add_argument("--text", action="append", help="Inline text snippet (can repeat).")
    parser.add_argument("--text-file", action="append", type=Path, help="Path to text file(s).")
    parser.add_argument("--audio", action="append", type=Path, help="Path to audio file(s) for ASR.")
    parser.add_argument("--images", type=Path, action="append", help="Image file or directory containing images.")
    parser.add_argument("--lang", help="Language hint for persona extraction (e.g., ja-JP, en-US).")
    parser.add_argument("--speaker", action="append", dest="speakers", help="Speaker labels to keep after diarisation.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum ASR confidence to keep.")
    parser.add_argument("--caption-model", default="Salesforce/blip-image-captioning-large", help="Caption model id.")
    parser.add_argument("--caption-device", default="cpu", help="Device for caption model.")
    parser.add_argument("--caption-max-tokens", type=int, default=32, help="Max tokens for captions.")
    parser.add_argument("--asr-model", default="medium", help="faster-whisper model size.")
    parser.add_argument("--asr-device", default="cpu", help="Device for faster-whisper.")
    parser.add_argument("--asr-compute-type", default="int8", help="Compute type (int8/int16/float16/float32).")
    parser.add_argument("--asr-beam-size", type=int, default=5, help="Beam size for ASR decoding.")
    parser.add_argument("--asr-language", help="Language hint passed to ASR.")
    parser.add_argument("--asr-vad-threshold", type=float, default=0.5, help="ASR VAD threshold.")
    parser.add_argument("--diarize", action="store_true", help="Enable diarisation for audio.")
    parser.add_argument("--diarize-model", help="Custom diarisation model id.")
    parser.add_argument("--diarize-token", help="Auth token for diarisation model.")
    parser.add_argument("--persona-out", type=Path, help="Path to write persona YAML.")
    parser.add_argument("--no-preview", action="store_true", help="Suppress YAML preview.")
    return parser.parse_args()


def load_text_snippets(args: argparse.Namespace) -> List[str]:
    snippets: List[str] = []
    if args.text:
        snippets.extend([s for s in args.text if s])
    if args.text_file:
        for path in args.text_file:
            if path.exists():
                snippets.append(path.read_text(encoding="utf-8"))
    return snippets


def collect_image_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if not args.images:
        return paths
    for entry in args.images:
        if not entry.exists():
            continue
        if entry.is_file():
            paths.append(entry)
        else:
            for sub in entry.iterdir():
                if sub.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    paths.append(sub)
    return paths


def collect_audio_segments(args: argparse.Namespace) -> List[SpeechSegment]:
    from emot_terrain_lab.audio.asr_pipeline import SpeechSegment  # local import to avoid circular

    collected: List[SpeechSegment] = []
    if not args.audio:
        return collected
    cfg = ASRConfig(
        model_size=args.asr_model,
        device=args.asr_device,
        compute_type=args.asr_compute_type,
        beam_size=args.asr_beam_size,
        language=args.asr_language,
        vad_threshold=args.asr_vad_threshold,
        diarization=args.diarize,
        diarization_model=args.diarize_model,
        diarization_auth_token=args.diarize_token,
    )
    transcriber = ASRTranscriber(cfg)
    for audio_path in args.audio:
        collected.extend(transcriber.transcribe(audio_path))
    return collected


def caption_images(args: argparse.Namespace) -> List[str]:
    image_paths = collect_image_paths(args)
    if not image_paths:
        return []
    cfg = CaptionConfig(
        model_id=args.caption_model,
        device=args.caption_device,
        max_new_tokens=args.caption_max_tokens,
    )
    captioner = ImageCaptioner(cfg)
    return captioner.caption(image_paths)


def main() -> None:
    args = parse_args()
    text_snippets = load_text_snippets(args)
    speech_segments = collect_audio_segments(args)
    image_captions = caption_images(args)

    builder = MultimodalPreferenceBuilder(
        speech_extractor=SpeechPreferenceExtractor(
            target_speakers=set(args.speakers) if args.speakers else None,
            min_confidence=args.min_confidence,
        ),
    )
    persona = builder.build(
        text_chunks=text_snippets,
        speech_segments=speech_segments,
        image_captions=image_captions,
        lang_hint=args.lang,
    )

    if not args.no_preview:
        print(persona.to_yaml())
        if persona.notes:
            print("# notes:")
            for note in persona.notes:
                print(f"# - {note}")

    if args.persona_out:
        persona.save(args.persona_out)


if __name__ == "__main__":
    main()
