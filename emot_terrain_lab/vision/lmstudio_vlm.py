from __future__ import annotations

from dataclasses import dataclass
import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from terrain import llm as terrain_llm


def _env_flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {"1", "true", "on", "yes"}


@dataclass
class LMStudioVLMConfig:
    enabled: bool = False
    max_tokens: int = 180
    temperature: float = 0.1
    request_timeout: float = 120.0
    retries: int = 2
    retry_delay_seconds: float = 1.0
    system_prompt: str = (
        "You summarize an observed scene for an emotionally aware companion runtime. "
        "Describe only grounded visible details, social cues, atmosphere, and uncertainty. "
        "Keep it concise."
    )

    @classmethod
    def from_env(cls) -> "LMStudioVLMConfig":
        return cls(
            enabled=_env_flag("EQNET_VLM_ENABLED", os.getenv("USE_VLM", "0")),
            max_tokens=int(os.getenv("EQNET_VLM_MAX_TOKENS", "180")),
            temperature=float(os.getenv("EQNET_VLM_TEMPERATURE", "0.1")),
            request_timeout=float(os.getenv("EQNET_VLM_TIMEOUT", "120")),
            retries=int(os.getenv("EQNET_VLM_RETRIES", "2")),
            retry_delay_seconds=float(os.getenv("EQNET_VLM_RETRY_DELAY", "1.0")),
            system_prompt=os.getenv(
                "EQNET_VLM_SYSTEM_PROMPT",
                "You summarize an observed scene for an emotionally aware companion runtime. "
                "Describe only grounded visible details, social cues, atmosphere, and uncertainty. "
                "Keep it concise.",
            ),
        )


class LMStudioVLMAdapter:
    def __init__(self, config: Optional[LMStudioVLMConfig] = None) -> None:
        self.config = config or LMStudioVLMConfig.from_env()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _to_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        mime, _ = mimetypes.guess_type(path.name)
        if not mime:
            mime = "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        outputs = payload.get("output")
        if not isinstance(outputs, list):
            return ""
        chunks: list[str] = []
        for item in outputs:
            if not isinstance(item, dict):
                continue
            contents = item.get("content")
            if not isinstance(contents, list):
                continue
            for content in contents:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
        return "\n".join(chunks).strip()

    def _request_payload(self, llm: Any, *, image_path: str, prompt: str) -> Dict[str, Any]:
        data_url = self._to_data_url(image_path)
        return {
            "model": llm.model,
            "instructions": self.config.system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            "max_output_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

    def _responses_endpoint(self, base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized + "/responses"
        return normalized + "/v1/responses"

    def _post_with_retry(self, endpoint: str, *, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        attempts = max(1, int(self.config.retries) + 1)
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.config.request_timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                should_retry = attempt < attempts
                if isinstance(exc, requests.HTTPError):
                    status = exc.response.status_code if exc.response is not None else None
                    should_retry = should_retry and status is not None and status >= 500
                if not should_retry:
                    raise
                time.sleep(max(0.0, self.config.retry_delay_seconds) * attempt)
        if last_error is not None:
            raise last_error
        raise RuntimeError("LM Studio VLM request failed without an exception")

    def describe_image(self, image_path: str, *, user_text: Optional[str] = None) -> Dict[str, Any]:
        path = Path(image_path)
        if not self.enabled or not path.exists():
            return {}
        llm = terrain_llm.get_llm()
        if not llm.available:
            return {}
        prompt = user_text or "Summarize the visible scene for context."
        endpoint = self._responses_endpoint(llm.base_url)
        payload = self._request_payload(llm, image_path=str(path), prompt=prompt)
        headers = {
            "Authorization": f"Bearer {llm.api_key}",
            "Content-Type": "application/json",
        }
        body = self._post_with_retry(endpoint, payload=payload, headers=headers)
        text = self._extract_text(body)
        if not text:
            return {}
        return {
            "backend": "lmstudio_vlm",
            "model": llm.model,
            "image_path": str(path),
            "text": text,
            "response_id": body.get("id"),
        }
