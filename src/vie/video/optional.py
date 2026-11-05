"""The optional layers: captions and speech. Off by default; each is one flag.

Speech is often where a video's actual subject lives. A news clip of two people at
a desk looks identical whatever they are discussing, so transcription is what turns
"a news broadcast" into something you can search for by topic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vie.config import Config
from vie.logging_setup import get_logger
from vie.video.understand import TranscriptSegment

log = get_logger("video.optional")


class WhisperTranscriber:
    """Speech to text with faster-whisper, which reads the audio via PyAV itself."""

    def __init__(self, config: Config) -> None:
        self.model_name = config.video.whisper_model
        self.device = config.video.whisper_device
        self.keep_loaded = not config.low_memory
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.device
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute)
            log.info("loaded whisper %s on %s", self.model_name, device)
        return self._model

    def transcribe(
        self, path: str, duration: float
    ) -> tuple[list[TranscriptSegment], str | None]:
        """Transcribe up to ``duration`` seconds, the span the frames were analysed for.

        Samples are decoded here and handed over as an array. Left to read the file
        itself, faster-whisper 1.2 calls ``av.open`` with an option current PyAV has
        removed, and it cannot be pinned around because older PyAV has no wheels for
        current Python on Windows.
        """
        from vie.video.decode import read_audio

        audio = read_audio(path, duration)
        if not audio.size:
            return [], None
        segments, info = self._load().transcribe(audio, vad_filter=True)
        # Segments are lazy: the model only does its work while they are consumed.
        found = [
            TranscriptSegment(float(s.start), float(s.end), s.text.strip())
            for s in segments
            if s.text.strip()
        ]
        if not self.keep_loaded:
            self._model = None
        return found, info.language if found else None


class BlipCaptioner:
    """One sentence per shot. Reads well, and can be wrong: shown as model output."""

    def __init__(self, config: Config, runtime: Any) -> None:
        self.model_name = config.video.caption_model
        self.runtime = runtime
        self._loaded: tuple[Any, Any] | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._loaded is None:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            processor = BlipProcessor.from_pretrained(self.model_name)
            model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self._loaded = (processor, model.to(self.runtime.device).eval())
            log.info("loaded %s", self.model_name)
        return self._loaded

    def caption(self, images: Sequence[Any]) -> list[str]:
        import torch

        from vie.models import bgr_to_pil

        processor, model = self._load()
        captions: list[str] = []
        for start in range(0, len(images), 4):
            chunk = [bgr_to_pil(image) for image in images[start:start + 4]]
            inputs = processor(images=chunk, return_tensors="pt").to(self.runtime.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=30, num_beams=3)
            captions += [processor.decode(o, skip_special_tokens=True).strip() for o in output]
        return captions
