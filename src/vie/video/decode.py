"""Video decoding on PyAV.

PyAV rather than ``cv2.VideoCapture`` for three concrete reasons:

* **Real timestamps.** Phone footage is variable frame rate, so ``frame_index / fps``
  drifts by seconds over a clip. Every result here is reported as a timestamp a
  person will seek to, so it has to come from the container.
* **Non-ASCII paths.** ``VideoCapture`` fails on them on Windows.
* **Rotation.** A portrait phone video is stored landscape with a rotation flag.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vie.logging_setup import get_logger

log = get_logger("video.decode")

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts"}


class VideoDecodeError(RuntimeError):
    """The file could not be opened or holds no decodable video."""


@dataclass(frozen=True)
class VideoInfo:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    codec: str
    has_audio: bool


@dataclass
class Frame:
    time: float
    image: np.ndarray  # BGR, the same convention as the image pipeline


def _open(path: str | Path) -> Any:
    import av

    if not Path(path).is_file():
        raise FileNotFoundError(f"video not found: {path}")
    try:
        container = av.open(str(path))
    except (av.FFmpegError, OSError) as exc:
        raise VideoDecodeError(f"could not open {path}: {exc}") from exc
    if not container.streams.video:
        container.close()
        raise VideoDecodeError(f"{path} contains no video stream")
    return container


def probe(path: str | Path) -> VideoInfo:
    """Read container metadata without decoding any frames."""
    container = _open(path)
    try:
        stream = container.streams.video[0]
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        rate = stream.average_rate or stream.guessed_rate
        return VideoInfo(
            path=str(path),
            duration=duration,
            fps=float(rate) if rate else 0.0,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            codec=stream.codec_context.name,
            has_audio=bool(container.streams.audio),
        )
    finally:
        container.close()


def _frame_time(frame: Any, stream: Any, fallback: float) -> float:
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    return fallback


def _to_bgr(frame: Any, max_side: int) -> np.ndarray:
    """Convert to BGR, honouring rotation and capping the longest side."""
    width, height = frame.width, frame.height
    scale = min(1.0, max_side / max(width, height)) if max_side else 1.0
    if scale < 1.0:
        # Even dimensions keep every pixel format's chroma planes valid.
        width = max(2, int(width * scale) // 2 * 2)
        height = max(2, int(height * scale) // 2 * 2)
        frame = frame.reformat(width=width, height=height)
    image = frame.to_ndarray(format="bgr24")

    rotation = int(getattr(frame, "rotation", 0) or 0) % 360
    if rotation:
        # The flag says how far to rotate counter-clockwise for display.
        image = np.rot90(image, k=rotation // 90).copy()
    return image


def iter_decoded(path: str | Path, max_side: int = 0) -> Iterator[Frame]:
    """Yield every decodable frame in order, tolerating a truncated file."""
    import av

    container = _open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        last = 0.0
        try:
            for packet in container.demux(stream):
                for frame in packet.decode():
                    last = _frame_time(frame, stream, last)
                    yield Frame(last, _to_bgr(frame, max_side))
        except av.FFmpegError as exc:
            # A damaged tail still leaves everything before it usable.
            log.warning("stopped decoding %s at %.1f s: %s", path, last, exc)
    finally:
        container.close()


def sample_frames(
    path: str | Path, every: float, max_side: int = 0, max_duration: float | None = None
) -> Iterator[Frame]:
    """Yield one frame per ``every`` seconds of *video time*."""
    if every <= 0:
        raise ValueError("sampling interval must be positive")
    next_time = 0.0
    for frame in iter_decoded(path, max_side):
        if max_duration is not None and frame.time > max_duration:
            return
        if frame.time + 1e-6 >= next_time:
            yield frame
            next_time = frame.time + every


def frames_at(
    path: str | Path, times: Iterable[float], max_side: int = 0
) -> Iterator[Frame]:
    """Yield the first frame at or after each requested time.

    Seeks across long gaps, so a handful of keyframes from a long film does not
    pay for decoding all of it.
    """
    import av

    wanted = sorted(times)
    if not wanted:
        return

    container = _open(path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        position = 0
        last = 0.0
        needs_seek = False
        try:
            while position < len(wanted):
                target = wanted[position]
                if needs_seek and stream.time_base is not None:
                    container.seek(
                        int(max(0.0, target - 1.0) / stream.time_base),
                        stream=stream, backward=True,
                    )
                needs_seek = False
                restart = False
                for frame in container.decode(stream):
                    last = _frame_time(frame, stream, last)
                    while position < len(wanted) and last + 1e-6 >= wanted[position]:
                        yield Frame(last, _to_bgr(frame, max_side))
                        position += 1
                    if position >= len(wanted):
                        return
                    if wanted[position] - last > 10.0:
                        needs_seek = True
                        restart = True
                        break
                if not restart:
                    return  # reached the end of the stream
        except av.FFmpegError as exc:
            log.warning("stopped decoding %s at %.1f s: %s", path, last, exc)
    finally:
        container.close()


def read_audio(
    path: str | Path, duration: float | None = None, sample_rate: int = 16_000
) -> np.ndarray:
    """Decode the first audio track to mono float32 in [-1, 1]."""
    import av

    container = _open(path)
    try:
        if not container.streams.audio:
            return np.empty(0, dtype=np.float32)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        limit = None if duration is None else int(duration * sample_rate)
        chunks: list[np.ndarray] = []
        total = 0

        def keep(frames: list[Any]) -> None:
            nonlocal total
            for frame in frames:
                data = frame.to_ndarray().ravel()
                chunks.append(data)
                total += data.size

        try:
            for frame in container.decode(stream):
                keep(resampler.resample(frame))
                if limit is not None and total >= limit:
                    break
            else:
                keep(resampler.resample(None))
        except av.FFmpegError as exc:
            log.warning("stopped decoding audio of %s: %s", path, exc)
        if not chunks:
            return np.empty(0, dtype=np.float32)
        return (np.concatenate(chunks)[:limit].astype(np.float32) / 32768.0)
    finally:
        container.close()


def discover_videos(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"video folder not found: {root}")
    walker = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        p for p in walker
        if p.suffix.lower() in VIDEO_SUFFIXES
        and p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
