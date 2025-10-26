"""Vision→Markdown ingest interface (DeepSeek-OCR compatible).

Purpose
- Provide a stable entry point to inject structure-preserving Markdown derived
  from visual documents (PDF/images) into EQNet's RAG/knowledge layer.
- Actual OCR/LLM compression is delegated to a backend callable so that
  DeepSeek-OCR or any future model can be plugged in without changing call sites.

This module does not perform OCR itself. It defines:
- VisionToMarkdown: wrapper that calls a provided backend(file_path)->markdown
- simple stats helpers (approx token count, heading count)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Iterable, List, Sequence


MarkdownBackend = Callable[[Path], str]


@dataclass
class VisionDoc:
    doc_id: str
    markdown: str
    meta: Dict[str, float]


@dataclass
class VisionToMarkdownConfig:
    backend_name: str = "external"
    max_tokens_hint: int = 800  # target compression budget


class VisionToMarkdown:
    def __init__(self, backend: MarkdownBackend, config: Optional[VisionToMarkdownConfig] = None) -> None:
        self.backend = backend
        self.config = config or VisionToMarkdownConfig()

    def convert_file(self, path: Path) -> VisionDoc:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        md = self.backend(path)
        meta = self._md_stats(md)
        return VisionDoc(doc_id=path.stem, markdown=md, meta=meta)

    def batch(self, paths: Sequence[Path]) -> List[VisionDoc]:
        docs: List[VisionDoc] = []
        for p in paths:
            try:
                docs.append(self.convert_file(Path(p)))
            except Exception:
                # Skip unreadable paths; continue batch processing
                continue
        return docs

    @staticmethod
    def _md_stats(md: str) -> Dict[str, float]:
        # Shallow metrics to aid routing/indexing
        lines = md.splitlines()
        headings = sum(1 for ln in lines if ln.strip().startswith("#"))
        code_blocks = sum(1 for ln in lines if ln.strip().startswith("```")) // 2
        approx_tokens = len(md) / 4.0
        return {
            "md_lines": float(len(lines)),
            "md_headings": float(headings),
            "md_code_blocks": float(code_blocks),
            "md_tokens_approx": float(approx_tokens),
        }


def dummy_backend(path: Path) -> str:
    """Placeholder backend when no OCR is available.

    Produces a minimal front-matter indicating the file name and a stub note.
    Replace this with a DeepSeek-OCR wrapper that returns structured Markdown.
    """
    name = path.name
    return f"""# Document: {name}

> NOTE: Markdown content is not available in this environment.
> Replace `dummy_backend` with a DeepSeek‑OCR powered backend.

"""


__all__ = ["VisionToMarkdown", "VisionToMarkdownConfig", "VisionDoc", "dummy_backend"]
