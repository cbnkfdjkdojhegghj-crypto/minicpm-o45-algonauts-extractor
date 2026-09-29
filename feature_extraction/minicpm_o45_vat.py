#!/usr/bin/env python3
"""Frozen MiniCPM-o 4.5 VAT feature extraction over streamed Algonauts stimuli.

This implementation intentionally contains no fMRI loading, HRF shifting, PCA,
brain readout, subject alignment, or downstream modelling.

Engineering choices mirror the released BrainBridge VAT extractor where that
code is available: movie-level decode, TR-indexed context windows, overlapping
transcript selection, proportional valid-token pooling, mean+sample-std
statistics, atomic outputs, completion markers, and resumable streaming.

The original MiniCPM-specific BrainBridge extractor referenced by the release
was not included in release v1.0.2. MiniCPM input packing below therefore uses
the official MiniCPM-o 4.5 processor/API directly.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.streaming_stimuli import (  # noqa: E402
    Batch,
    annex_drop,
    annex_get,
    batch_paths,
    build_index,
    enable_dataset,
    make_batches,
)


MODEL_ID = "openbmb/MiniCPM-o-4_5"
# Pinned current upstream revision, resolved from Hugging Face on 2026-09-29.
MODEL_REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
DEFAULT_PAPER_LAYER = 18


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class Window:
    sample_index: int
    chunk_start: float
    chunk_end: float
    rel_start: float
    rel_end: float


def build_windows(
    *,
    target_sample_number: int,
    video_duration: float,
    tr: float,
    chunk_length: float,
    seconds_before_chunk: float,
) -> list[Window]:
    if target_sample_number < 1:
        raise ValueError("target_sample_number must be positive")
    if not (0 < tr <= chunk_length):
        raise ValueError("TR must be positive and no longer than chunk_length")
    if seconds_before_chunk < 0 or seconds_before_chunk + tr > chunk_length:
        raise ValueError("invalid context/chunk geometry")
    if video_duration <= 0:
        raise ValueError("video_duration must be positive")

    max_interest_start = max(0.0, video_duration - tr)
    max_chunk_start = max(0.0, video_duration - chunk_length)
    windows: list[Window] = []
    for sample_index in range(target_sample_number):
        interest_start = min(sample_index * tr, max_interest_start)
        chunk_start = max(0.0, min(interest_start - seconds_before_chunk, max_chunk_start))
        chunk_end = min(chunk_start + chunk_length, video_duration)
        rel_start = interest_start - chunk_start
        rel_end = min(rel_start + tr, chunk_end - chunk_start)
        windows.append(Window(sample_index, chunk_start, chunk_end, rel_start, rel_end))
    return windows


@dataclass
class TranscriptIndex:
    starts: np.ndarray
    ends: np.ndarray
    texts: list[str]

    @classmethod
    def empty(cls) -> "TranscriptIndex":
        return cls(np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), [])

    @classmethod
    def from_tsv(cls, path: Path | None) -> "TranscriptIndex":
        if path is None or not path.is_file():
            return cls.empty()

        import pandas as pd

        frame = pd.read_csv(
            path,
            sep="\t",
            skiprows=1,
            header=None,
            names=["text", "words", "onsets", "durations"],
            on_bad_lines="skip",
            keep_default_na=False,
        )
        starts: list[float] = []
        ends: list[float] = []
        texts: list[str] = []
        for row in frame.itertuples(index=False):
            try:
                onsets = ast.literal_eval(str(row.onsets))
                durations = ast.literal_eval(str(row.durations))
                if not onsets or not durations:
                    continue
                start = float(onsets[0])
                end = float(onsets[-1]) + float(durations[-1])
                text = str(row.text).replace("\n", " ").replace("\r", " ").strip()
                if text and math.isfinite(start) and math.isfinite(end) and end > start:
                    starts.append(start)
                    ends.append(end)
                    texts.append(text)
            except (SyntaxError, ValueError, TypeError, IndexError):
                continue
        return cls(np.asarray(starts), np.asarray(ends), texts)

    def slice(self, start: float, end: float) -> str:
        if not self.texts:
            return ""
        mask = (self.starts < end) & (self.ends > start)
        return " ".join(text for text, include in zip(self.texts, mask.tolist()) if include)


@dataclass
class MovieMedia:
    frames: np.ndarray
    audio: np.ndarray
    transcript: TranscriptIndex
    duration: float
    video_sample_fps: float
    audio_sample_rate: int
    decoded_height: int


def probe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream: {path}")
    stream = streams[0]
    duration = float(stream.get("duration") or (payload.get("format") or {}).get("duration"))
    width, height = int(stream["width"]), int(stream["height"])
    if duration <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video metadata for {path}")
    return {"duration": duration, "width": width, "height": height}


def even_scaled_height(source_width: int, source_height: int, target_width: int) -> int:
    return max(2, int(round(source_height * target_width / source_width / 2.0)) * 2)


def compute_video_sample_fps(*, tr: float, chunk_length: float, frames_per_tr: int) -> float:
    if tr <= 0 or chunk_length <= 0 or frames_per_tr < 1:
        raise ValueError("invalid video sampling parameters")
    frame_count = max(1, int(chunk_length / tr * frames_per_tr))
    return frame_count / chunk_length


def decode_video_ffmpeg(
    path: Path,
    *,
    sample_fps: float,
    resize_width: int,
    ffmpeg_threads: int,
) -> tuple[np.ndarray, float, int]:
    metadata = probe_video(path)
    height = even_scaled_height(metadata["width"], metadata["height"], resize_width)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        str(max(1, ffmpeg_threads)),
        "-i",
        str(path),
        "-an",
        "-vf",
        f"fps={sample_fps:.12f},scale={resize_width}:{height}:flags=bicubic",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    timeout = max(180.0, metadata["duration"] * 2.0)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"ffmpeg video decode failed for {path}: {message}")
    frame_bytes = resize_width * height * 3
    if not result.stdout or len(result.stdout) % frame_bytes:
        raise RuntimeError(f"unexpected raw video byte count for {path}")
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, resize_width, 3)
    if len(frames) < 1:
        raise RuntimeError(f"decoded no frames from {path}")
    return frames, float(metadata["duration"]), height


def decode_audio_ffmpeg(path: Path, *, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"ffmpeg audio decode failed for {path}: {message}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"decoded empty audio from {path}")
    audio = np.nan_to_num(audio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_movie_media(
    movie_path: Path,
    transcript_path: Path | None,
    *,
    video_sample_fps: float,
    resize_width: int,
    audio_sample_rate: int,
    ffmpeg_threads: int,
) -> MovieMedia:
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="minicpm-media") as executor:
        video_future = executor.submit(
            decode_video_ffmpeg,
            movie_path,
            sample_fps=video_sample_fps,
            resize_width=resize_width,
            ffmpeg_threads=ffmpeg_threads,
        )
        audio_future = executor.submit(decode_audio_ffmpeg, movie_path, sample_rate=audio_sample_rate)
        transcript = TranscriptIndex.from_tsv(transcript_path)
        frames, duration, decoded_height = video_future.result()
        audio = audio_future.result()
    return MovieMedia(
        frames=frames,
        audio=audio,
        transcript=transcript,
        duration=duration,
        video_sample_fps=video_sample_fps,
        audio_sample_rate=audio_sample_rate,
        decoded_height=decoded_height,
    )


def fixed_frame_count(chunk_length: float, sample_fps: float) -> int:
    return max(1, int(round(chunk_length * sample_fps)))


def slice_window_inputs(
    media: MovieMedia,
    window: Window,
    *,
    chunk_length: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    frame_count = fixed_frame_count(chunk_length, media.video_sample_fps)
    first_frame = int(round(window.chunk_start * media.video_sample_fps))
    frame_indices = np.arange(first_frame, first_frame + frame_count, dtype=np.int64)
    frame_indices = np.clip(frame_indices, 0, len(media.frames) - 1)
    frames = np.ascontiguousarray(media.frames[frame_indices])

    audio_count = int(round(chunk_length * media.audio_sample_rate))
    audio_start = int(round(window.chunk_start * media.audio_sample_rate))
    audio_end = min(audio_start + audio_count, len(media.audio))
    audio = np.zeros(audio_count, dtype=np.float32)
    available = max(0, audio_end - audio_start)
    if available:
        audio[:available] = media.audio[audio_start:audio_end]

    text = media.transcript.slice(window.chunk_start, window.chunk_end)
    return frames, audio, text


def make_prompt(movie_id: str, transcript: str) -> str:
    if transcript:
        return (
            f"Clip id: {movie_id}. Dialogue transcript: '{transcript}' "
            "Please comprehend all modalities."
        )
    return (
        f"Clip id: {movie_id}. There is no dialogue transcript for this segment. "
        "Please comprehend all modalities."
    )


def load_model(
    *,
    device: str,
    model_id: str,
    revision: str,
    attn_implementation: str,
):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    model.prepare_processor()
    return model, model.processor


def llm_layers(model: Any) -> Sequence[Any]:
    layers = model.llm.model.layers
    if len(layers) < DEFAULT_PAPER_LAYER:
        raise RuntimeError(f"unexpected MiniCPM thinker depth: {len(layers)}")
    return layers


class LayerCapture:
    def __init__(self, model: Any, paper_layers: Sequence[int]) -> None:
        self.paper_layers = tuple(int(x) for x in paper_layers)
        self.states: dict[int, Any] = {}
        self.handles = []
        layers = llm_layers(model)
        for paper_layer in self.paper_layers:
            if paper_layer < 1 or paper_layer > len(layers):
                raise ValueError(f"paper layer {paper_layer} outside 1..{len(layers)}")
            module = layers[paper_layer - 1]

            def hook(_module: Any, _inputs: Any, output: Any, *, layer=paper_layer) -> None:
                value = output[0] if isinstance(output, tuple) else output
                self.states[layer] = value.detach()

            self.handles.append(module.register_forward_hook(hook))

    def clear(self) -> None:
        self.states.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def prepare_minicpm_batch(
    *,
    model: Any,
    processor: Any,
    movie_id: str,
    media: MovieMedia,
    windows: Sequence[Window],
    chunk_length: float,
    max_input_length: int,
    max_slice_nums: int,
):
    from PIL import Image

    prompts: list[str] = []
    input_images: list[list[Any]] = []
    input_audios: list[list[np.ndarray]] = []
    audio_parts: list[list[int]] = []

    for window in windows:
        frames, audio, transcript = slice_window_inputs(media, window, chunk_length=chunk_length)
        images = [Image.fromarray(frame, mode="RGB") for frame in frames]
        prompt_text = make_prompt(movie_id, transcript)

        # Mirrors MiniCPM-o 4.5's public chat preprocessing without generation:
        # PIL images -> <image>, ndarray audio -> <audio>, then apply chat template.
        message_content = ["<image>./</image>"] * len(images)
        message_content.append("<audio>./</audio>")
        message_content.append(prompt_text)
        messages = [{"role": "user", "content": "\n".join(message_content)}]
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            use_tts_template=True,
            enable_thinking=False,
        )

        prompts.append(prompt)
        input_images.append(images)
        input_audios.append([audio])
        audio_parts.append([0])

    batch = processor(
        prompts,
        input_images,
        input_audios,
        audio_parts,
        max_slice_nums=max_slice_nums,
        use_image_id=False,
        stream_input=False,
        return_tensors="pt",
        max_length=max_input_length,
    ).to(model.device)
    batch.pop("image_sizes", None)
    return batch


def proportional_pool(
    states: Any,
    attention_mask: Any,
    windows: Sequence[Window],
    *,
    chunk_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if states.ndim != 3:
        raise ValueError(f"expected [batch,sequence,width], got {tuple(states.shape)}")
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(states.shape[:2]):
        raise ValueError(
            f"attention mask {tuple(attention_mask.shape)} incompatible with states {tuple(states.shape)}"
        )

    means: list[Any] = []
    stds: list[Any] = []
    values = states.float()
    for batch_index, window in enumerate(windows):
        positions = torch.nonzero(attention_mask[batch_index] != 0, as_tuple=False).flatten()
        token_count = int(positions.numel())
        if token_count == 0:
            raise RuntimeError("processor produced an empty token sequence")

        start = min(token_count - 1, int(round(window.rel_start / chunk_length * token_count)))
        end = min(
            token_count,
            max(start + 1, int(round(window.rel_end / chunk_length * token_count))),
        )
        selected_positions = positions[start:end]
        selected = values[batch_index].index_select(0, selected_positions)
        mean = selected.mean(dim=0)
        if selected.shape[0] > 1:
            std = selected.std(dim=0, correction=1)
        else:
            std = torch.zeros_like(mean)
        means.append(mean)
        stds.append(std)

    mean_np = torch.stack(means).to(dtype=torch.float16).cpu().numpy()
    std_np = torch.stack(stds).to(dtype=torch.float16).cpu().numpy()
    return mean_np, std_np


def batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def completion_marker(output_root: Path, movie_id: str) -> Path:
    return output_root / "_complete" / f"{movie_id}.json"


def output_path(output_root: Path, paper_layer: int, statistic: str, movie_id: str) -> Path:
    return output_root / f"layer_{paper_layer:02d}" / statistic / f"{movie_id}.npy"


def validate_outputs(
    *,
    output_root: Path,
    movie_id: str,
    sample_count: int,
    hidden_dim: int,
    paper_layers: Sequence[int],
    dtype: str,
    model_id: str,
    model_revision: str,
) -> tuple[bool, str]:
    marker = completion_marker(output_root, movie_id)
    if not marker.is_file():
        return False, "completion marker missing"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid completion marker: {exc}"

    expected_marker = {
        "state": "complete",
        "movie_id": movie_id,
        "sample_count": int(sample_count),
        "hidden_dim": int(hidden_dim),
        "dtype": np.dtype(dtype).name,
        "model_id": model_id,
        "model_revision": model_revision,
        "paper_layers": [int(x) for x in paper_layers],
    }
    for key, expected in expected_marker.items():
        if payload.get(key) != expected:
            return False, f"marker {key}={payload.get(key)!r} != {expected!r}"

    for layer in paper_layers:
        for statistic in ("mean", "std"):
            path = output_path(output_root, layer, statistic, movie_id)
            if not path.is_file():
                return False, f"missing {path}"
            array = np.load(path, mmap_mode="r")
            if array.shape != (sample_count, hidden_dim):
                return False, f"shape mismatch {path}: {array.shape}"
            if array.dtype != np.dtype(dtype):
                return False, f"dtype mismatch {path}: {array.dtype}"
    return True, "complete"


def load_target_counts(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"movie_id", "target_sample_number"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"target sample manifest missing columns: {sorted(missing)}")
    return {
        str(row.movie_id).lower(): int(row.target_sample_number)
        for row in frame.itertuples(index=False)
    }


def process_movie(
    *,
    model: Any,
    processor: Any,
    capture: LayerCapture,
    movie_id: str,
    movie_path: Path,
    transcript_path: Path | None,
    output_root: Path,
    target_sample_number: int | None,
    tr: float,
    chunk_length: float,
    seconds_before_chunk: float,
    frames_per_tr: int,
    resize_width: int,
    audio_sample_rate: int,
    ffmpeg_threads: int,
    inference_batch_size: int,
    max_input_length: int,
    max_slice_nums: int,
    storage_dtype: str,
    diagnostic_token_states: int,
    model_id: str,
    model_revision: str,
) -> None:
    import torch
    import transformers

    started = time.monotonic()
    metadata = probe_video(movie_path)
    if target_sample_number is None:
        target_sample_number = max(1, int(math.floor(metadata["duration"] / tr)))

    video_sample_fps = compute_video_sample_fps(
        tr=tr,
        chunk_length=chunk_length,
        frames_per_tr=frames_per_tr,
    )
    media = load_movie_media(
        movie_path,
        transcript_path,
        video_sample_fps=video_sample_fps,
        resize_width=resize_width,
        audio_sample_rate=audio_sample_rate,
        ffmpeg_threads=ffmpeg_threads,
    )
    windows = build_windows(
        target_sample_number=target_sample_number,
        video_duration=media.duration,
        tr=tr,
        chunk_length=chunk_length,
        seconds_before_chunk=seconds_before_chunk,
    )

    hidden_dim = int(model.llm.config.hidden_size)
    valid, reason = validate_outputs(
        output_root=output_root,
        movie_id=movie_id,
        sample_count=target_sample_number,
        hidden_dim=hidden_dim,
        paper_layers=capture.paper_layers,
        dtype=storage_dtype,
        model_id=model_id,
        model_revision=model_revision,
    )
    if valid:
        print(f"[skip] {movie_id}: {reason}", flush=True)
        return

    print(
        f"[extract] {movie_id}: samples={target_sample_number} duration={media.duration:.2f}s "
        f"fps={video_sample_fps:.4f} layers={capture.paper_layers}",
        flush=True,
    )

    arrays: dict[tuple[int, str], np.ndarray] = {}
    for layer in capture.paper_layers:
        arrays[(layer, "mean")] = np.empty((target_sample_number, hidden_dim), dtype=storage_dtype)
        arrays[(layer, "std")] = np.empty((target_sample_number, hidden_dim), dtype=storage_dtype)

    diagnostics_dir = output_root / "diagnostics" / movie_id
    diagnostics_saved = 0

    if str(model.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(model.device)

    for window_batch in batched(windows, inference_batch_size):
        capture.clear()
        batch = prepare_minicpm_batch(
            model=model,
            processor=processor,
            movie_id=movie_id,
            media=media,
            windows=window_batch,
            chunk_length=chunk_length,
            max_input_length=max_input_length,
            max_slice_nums=max_slice_nums,
        )
        attention_mask = batch["attention_mask"]

        with torch.inference_mode():
            # Reuse MiniCPM's official multimodal embedding path, but stop at
            # the base Qwen3 decoder so no unnecessary LM-head logits are built.
            thinker_inputs, _ = model.get_vllm_embedding(batch)
            thinker_inputs = model.get_omni_embedding(
                batch,
                input_embeddings=thinker_inputs,
                chunk_length=model.config.audio_chunk_length,
            )
            position_ids = batch["position_ids"]
            if position_ids.dtype != torch.int64:
                position_ids = position_ids.long()
            _ = model.llm.model(
                input_ids=None,
                inputs_embeds=thinker_inputs,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

        missing = [layer for layer in capture.paper_layers if layer not in capture.states]
        if missing:
            raise RuntimeError(f"forward hooks did not fire for layers {missing}")

        for layer in capture.paper_layers:
            states = capture.states[layer]
            mean, std = proportional_pool(
                states,
                attention_mask,
                window_batch,
                chunk_length=chunk_length,
            )
            indices = [window.sample_index for window in window_batch]
            arrays[(layer, "mean")][indices] = mean.astype(storage_dtype, copy=False)
            arrays[(layer, "std")][indices] = std.astype(storage_dtype, copy=False)

            if diagnostic_token_states > 0 and diagnostics_saved < diagnostic_token_states:
                for local_index, window in enumerate(window_batch):
                    if diagnostics_saved >= diagnostic_token_states:
                        break
                    positions = torch.nonzero(
                        attention_mask[local_index] != 0, as_tuple=False
                    ).flatten()
                    token_count = int(positions.numel())
                    start = min(
                        token_count - 1,
                        int(round(window.rel_start / chunk_length * token_count)),
                    )
                    end = min(
                        token_count,
                        max(
                            start + 1,
                            int(round(window.rel_end / chunk_length * token_count)),
                        ),
                    )
                    selected_positions = positions[start:end]
                    selected = (
                        states[local_index]
                        .index_select(0, selected_positions)
                        .to(dtype=torch.float16)
                        .cpu()
                        .numpy()
                    )
                    diagnostics_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        diagnostics_dir / f"tr_{window.sample_index:06d}_layer_{layer:02d}.npz",
                        hidden=selected,
                        valid_positions=positions.cpu().numpy(),
                        pooled_positions=selected_positions.cpu().numpy(),
                        rel_start=np.float32(window.rel_start),
                        rel_end=np.float32(window.rel_end),
                        chunk_start=np.float32(window.chunk_start),
                        chunk_end=np.float32(window.chunk_end),
                    )
                    diagnostics_saved += 1

        del batch
        capture.clear()

    for (layer, statistic), array in arrays.items():
        atomic_npy(output_path(output_root, layer, statistic, movie_id), array)

    elapsed = time.monotonic() - started
    peak_gpu_bytes = 0
    if str(model.device).startswith("cuda"):
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated(model.device))

    marker = {
        "schema": "minicpm-o-4.5-vat.v1",
        "state": "complete",
        "completed_at": utc_now(),
        "movie_id": movie_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "paper_layers": [int(x) for x in capture.paper_layers],
        "hook_semantics": "output of model.llm.model.layers[paper_layer-1]",
        "sample_count": int(target_sample_number),
        "hidden_dim": hidden_dim,
        "dtype": np.dtype(storage_dtype).name,
        "tr": tr,
        "chunk_length": chunk_length,
        "seconds_before_chunk": seconds_before_chunk,
        "frames_per_tr": frames_per_tr,
        "video_sample_fps": video_sample_fps,
        "resize_width": resize_width,
        "decoded_height": media.decoded_height,
        "audio_sample_rate": audio_sample_rate,
        "pooling": ["mean", "sample_std"],
        "pooling_alignment": "proportional_valid_token_crop",
        "pca": False,
        "frozen": True,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "movie_duration_seconds": media.duration,
        "elapsed_seconds": round(elapsed, 6),
        "peak_gpu_bytes": peak_gpu_bytes,
    }
    atomic_json(completion_marker(output_root, movie_id), marker)
    print(f"[done] {movie_id}: {elapsed / 60.0:.2f} min", flush=True)


def parse_layers(value: str) -> list[int]:
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("layers must be unique")
    return layers


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def cfg_get(config: dict[str, Any], path: str, default: Any) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "minicpm_o45_vat.yaml")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-branch", default="main")
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--annex-jobs", type=int, default=8)
    parser.add_argument("--keep-downloaded", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=parse_layers)
    parser.add_argument("--target-samples-manifest", type=Path)
    parser.add_argument("--diagnostic-token-states", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)

    model_id = str(cfg_get(config, "model.id", MODEL_ID))
    model_revision = str(cfg_get(config, "model.revision", MODEL_REVISION))
    attn_implementation = str(cfg_get(config, "model.attn_implementation", "sdpa"))

    tr = float(cfg_get(config, "temporal.tr", 1.49))
    chunk_length = float(cfg_get(config, "temporal.chunk_length", 8.0))
    seconds_before_chunk = float(cfg_get(config, "temporal.seconds_before_chunk", 4.0))

    frames_per_tr = int(cfg_get(config, "video.frames_per_tr", 2))
    resize_width = int(cfg_get(config, "video.resize_width", 448))
    ffmpeg_threads = int(cfg_get(config, "video.ffmpeg_threads", 4))
    audio_sample_rate = int(cfg_get(config, "audio.sample_rate", 16000))

    inference_batch_size = int(cfg_get(config, "runtime.inference_batch_size", 1))
    cpu_threads = int(cfg_get(config, "runtime.cpu_threads", 2))
    cpu_interop_threads = int(cfg_get(config, "runtime.cpu_interop_threads", 1))
    max_input_length = int(cfg_get(config, "runtime.max_input_length", 8192))
    max_slice_nums = int(cfg_get(config, "runtime.max_slice_nums", 1))
    storage_dtype = str(cfg_get(config, "storage.dtype", "float16"))

    layers = args.layers or [
        int(x) for x in cfg_get(config, "representation.production_layers", [DEFAULT_PAPER_LAYER])
    ]
    diagnostic_token_states = (
        args.diagnostic_token_states
        if args.diagnostic_token_states is not None
        else int(cfg_get(config, "diagnostics.token_states_per_movie", 0))
    )

    if args.batch_size < 1 or args.annex_jobs < 1 or inference_batch_size < 1:
        raise SystemExit("batch-size, annex-jobs and inference-batch-size must be positive")
    if cpu_threads < 1 or cpu_interop_threads < 1:
        raise SystemExit("cpu thread counts must be positive")
    if diagnostic_token_states < 0:
        raise SystemExit("diagnostic-token-states must be non-negative")

    # Keep the 4-core host from oversubscribing itself while FFmpeg, pandas,
    # tokenization, and PyTorch all compete for the same CPUs.
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))
    import torch
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(cpu_interop_threads)

    enable_dataset(args.dataset_root, args.dataset_branch)
    index = build_index(args.dataset_root)

    if args.episodes:
        episodes = [episode.lower() for episode in args.episodes]
    else:
        episodes = [
            episode
            for episode in index.episodes
            if episode in index.movies and episode in index.transcripts
        ]
    batches = make_batches(episodes, args.batch_size)
    end = len(batches) if args.max_batches is None else min(
        len(batches), args.start_batch + args.max_batches
    )

    target_counts = load_target_counts(args.target_samples_manifest)
    model, processor = load_model(
        device=args.device,
        model_id=model_id,
        revision=model_revision,
        attn_implementation=attn_implementation,
    )
    capture = LayerCapture(model, layers)
    print(
        f"MiniCPM loaded: model={model_id}@{model_revision[:12]} device={model.device} "
        f"thinker_layers={len(llm_layers(model))} hidden={model.llm.config.hidden_size} hooks={layers} "
        f"inference_batch={inference_batch_size} cpu_threads={cpu_threads}/{cpu_interop_threads}",
        flush=True,
    )

    try:
        for batch in batches[args.start_batch:end]:
            paths = batch_paths(
                index,
                batch,
                include_movie=True,
                include_transcript=True,
                require_all=True,
            )
            print(f"\n=== streaming batch {batch.index}: {batch.episodes} ===", flush=True)
            annex_get(paths, jobs=args.annex_jobs)
            success = False
            try:
                for movie_id in batch.episodes:
                    process_movie(
                        model=model,
                        processor=processor,
                        capture=capture,
                        movie_id=movie_id,
                        movie_path=index.movies[movie_id],
                        transcript_path=index.transcripts.get(movie_id),
                        output_root=args.output_root,
                        target_sample_number=target_counts.get(movie_id),
                        tr=tr,
                        chunk_length=chunk_length,
                        seconds_before_chunk=seconds_before_chunk,
                        frames_per_tr=frames_per_tr,
                        resize_width=resize_width,
                        audio_sample_rate=audio_sample_rate,
                        ffmpeg_threads=ffmpeg_threads,
                        inference_batch_size=inference_batch_size,
                        max_input_length=max_input_length,
                        max_slice_nums=max_slice_nums,
                        storage_dtype=storage_dtype,
                        diagnostic_token_states=diagnostic_token_states,
                        model_id=model_id,
                        model_revision=model_revision,
                    )
                success = True
            finally:
                if success and not args.keep_downloaded:
                    annex_drop(paths)
                elif not success:
                    print(
                        "Extraction failed; current streamed files were retained for inspection.",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        capture.close()


if __name__ == "__main__":
    main()
