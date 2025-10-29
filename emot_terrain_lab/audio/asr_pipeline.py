# -*- coding: utf-8 -*-
"""
Optional automatic speech recognition (ASR) + diarisation helpers.

The classes in this module intentionally wrap third-party packages lazily so
they do not add hard dependencies to the repository.  Install the following
packages if you plan to use the functionality:

    pip install faster-whisper  # ASR
    pip install pyannote.audio  # diarisation (optional)

The transcriber returns ``SpeechSegment`` objects which can be fed into the
persona preference extractor (see ``emot_terrain_lab.persona.speech_adapter``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class SpeechSegment:
    """An annotated speech fragment."""

    start: float
    end: float
    text: str
    confidence: float = 0.0
    speaker: str = "speaker_0"


@dataclass
class DiarizationTurn:
    start: float
    end: float
    speaker: str


@dataclass
class ASRConfig:
    model_size: str = "medium"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    language: Optional[str] = None
    vad_threshold: float = 0.5
    diarization: bool = False
    diarization_model: Optional[str] = None
    diarization_auth_token: Optional[str] = None


class Diarizer:
    """Thin wrapper around ``pyannote.audio`` diarisation."""

    def __init__(self, model_id: str, auth_token: Optional[str] = None) -> None:
        try:
            from pyannote.audio import Pipeline as PyannotePipeline  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "pyannote.audio is required for diarisation. Install it via "
                "`pip install pyannote.audio`."
            ) from exc
        kwargs = {}
        if auth_token:
            kwargs["use_auth_token"] = auth_token
        self._pipeline = PyannotePipeline.from_pretrained(model_id, **kwargs)

    def diarize(self, audio_path: Path) -> List[DiarizationTurn]:
        diarization = self._pipeline(str(audio_path))  # type: ignore[attr-defined]
        turns: List[DiarizationTurn] = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=str(speaker),
                )
            )
        return turns


class ASRTranscriber:
    """ASR wrapper using ``faster-whisper``."""

    def __init__(self, config: Optional[ASRConfig] = None) -> None:
        self.config = config or ASRConfig()
        self._model = None
        self._diarizer: Optional[Diarizer] = None
        if self.config.diarization:
            model_id = self.config.diarization_model or "pyannote/speaker-diarization"
            self._diarizer = Diarizer(model_id, auth_token=self.config.diarization_auth_token)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "faster-whisper is required for ASR. Install it via `pip install faster-whisper`."
            ) from exc
        self._model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    # ------------------------------------------------------------------ #

    def transcribe(self, audio_path: str | Path) -> List[SpeechSegment]:
        """
        Transcribe ``audio_path`` and return a list of speech segments.

        If diarisation is enabled (and ``pyannote.audio`` installed) the result
        will contain speaker labels, otherwise ``speaker`` defaults to
        ``"speaker_0"``.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file {path} does not exist.")

        self._ensure_model()
        assert self._model is not None

        transcribe_kwargs = {
            "beam_size": self.config.beam_size,
            "vad_filter": True,
            "vad_parameters": {"threshold": self.config.vad_threshold},
        }
        if self.config.language:
            transcribe_kwargs["language"] = self.config.language

        segments_iter, _ = self._model.transcribe(str(path), **transcribe_kwargs)
        raw_segments: List[SpeechSegment] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue
            # ``avg_logprob`` is negative; map to [0, 1] heuristically.
            confidence = float(max(0.0, min(1.0, 1.0 + seg.avg_logprob)))
            raw_segments.append(
                SpeechSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                    confidence=confidence,
                    speaker="speaker_0",
                )
            )

        if self._diarizer is None or not raw_segments:
            return raw_segments

        turns = self._diarizer.diarize(path)
        if not turns:
            return raw_segments

        return self._assign_speakers(raw_segments, turns)

    @staticmethod
    def _assign_speakers(
        segments: Iterable[SpeechSegment],
        turns: Iterable[DiarizationTurn],
    ) -> List[SpeechSegment]:
        turns_list = list(turns)
        assigned: List[SpeechSegment] = []
        for seg in segments:
            speaker = seg.speaker
            overlap_max = 0.0
            for turn in turns_list:
                overlap = _overlap(seg.start, seg.end, turn.start, turn.end)
                if overlap > overlap_max:
                    overlap_max = overlap
                    speaker = turn.speaker
            assigned.append(
                SpeechSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    confidence=seg.confidence,
                    speaker=speaker,
                )
            )
        return assigned


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


__all__ = [
    "SpeechSegment",
    "DiarizationTurn",
    "ASRConfig",
    "ASRTranscriber",
    "Diarizer",
]
