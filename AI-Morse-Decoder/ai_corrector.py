import os
from typing import AsyncIterator, Optional

import anthropic

SYSTEM_PROMPT = (
    "You are correcting morse code text decoded from audio. "
    "The decoding may contain errors from timing imperfections or interference. "
    "Domain: amateur (ham) radio communication.\n"
    "Common abbreviations: CQ (calling all stations), DE (from/this is), "
    "73 (best regards), 88 (love and kisses), QSL (confirmed), QTH (location), "
    "QRM (man-made interference), QSB (fading), QRN (static noise), "
    "QRZ (who is calling), QRP (low power), RST (readability-strength-tone), "
    "PSE (please), K (go ahead), AR (end of message), SK (end of contact), "
    "BK (break), HI (laughter), OM (old man), YL (young lady), ES (and), "
    "NR (number), UR (your), WX (weather), HPE (hope), TNX/TU (thanks).\n"
    "Output ONLY the corrected text. No explanations, no commentary."
)


class AICorrector:
    """Streams AI-corrected text for raw morse decode output via Claude API."""

    def __init__(self, model: str = "claude-opus-4-6", api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it or use --no-ai to disable AI correction."
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def correct(self, raw_text: str) -> AsyncIterator[str]:
        """Async generator that streams corrected tokens for raw_text."""
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Raw morse decode: {raw_text.strip()}"}
            ],
        ) as stream:
            async for token in stream.text_stream:
                yield token
