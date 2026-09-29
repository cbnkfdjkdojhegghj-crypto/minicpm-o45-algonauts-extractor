#!/usr/bin/env python3
"""Streaming access to Algonauts 2025 stimulus files via DataLad.

This module deliberately contains no feature-extraction logic. It only:
- installs/initialises the Algonauts competitors dataset using DataLad,
- discovers movie/transcript files by episode,
- materialises selected episode files with datalad get,
- and safely drops them after use with datalad drop.

Downstream extractors should import the functions here rather than copying the
dataset/download logic.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DATASET_URL = "https://github.com/courtois-neuromod/algonauts_2025.competitors.git"
DEFAULT_DATASET_BRANCH = "main"

EPISODE_RE = re.compile(
    r"(s\d{2}e\d{2}[a-z]?|(?:bourne|figures|life|wolf)\d{2}|"
    r"(?:chaplin|mononoke|passepartout|planetearth|pulpfiction|wot)\d{1,2})",
    re.IGNORECASE,
)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    printable = [re.sub(r"https://[^@]+@", "https://***@", str(x)) for x in cmd]
    print("+", " ".join(printable), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def run_optional(cmd: list[str], *, cwd: Path | None = None) -> bool:
    try:
        run(cmd, cwd=cwd)
        return True
    except subprocess.CalledProcessError as exc:
        print(
            f"WARNING: optional command failed with exit code {exc.returncode}; continuing: "
            + " ".join(str(x) for x in cmd),
            file=sys.stderr,
            flush=True,
        )
        return False


def require_commands() -> None:
    missing = [
        name
        for name in ("git", "git-annex", "datalad")
        if shutil.which(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing required commands: "
            + ", ".join(missing)
            + ". Install DataLad and git-annex before using streaming."
        )


def prepare_runtime() -> None:
    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    defaults = {
        "user.name": "Algonauts Streaming Runner",
        "user.email": "algonauts-streaming@localhost",
    }
    for key, value in defaults.items():
        result = subprocess.run(
            ["git", "config", "--global", "--get", key],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            run(["git", "config", "--global", key, value])


def ensure_git_identity(root: Path) -> None:
    defaults = {
        "user.name": "Algonauts Streaming Runner",
        "user.email": "algonauts-streaming@localhost",
    }
    for key, value in defaults.items():
        result = subprocess.run(
            ["git", "config", "--local", "--get", key],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            run(["git", "config", "--local", key, value], cwd=root)


def enable_dataset(
    dataset_root: Path,
    dataset_branch: str = DEFAULT_DATASET_BRANCH,
) -> None:
    """Install the Algonauts dataset recursively using DataLad."""
    require_commands()
    prepare_runtime()

    dataset_root = dataset_root.absolute()

    if not (dataset_root / ".git").exists():
        dataset_root.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "datalad",
                "install",
                "-r",
                "-s",
                DATASET_URL,
                str(dataset_root),
            ],
            cwd=dataset_root.parent,
        )

    ensure_git_identity(dataset_root)

    # Install registered subdatasets recursively without fetching content.
    run(
        [
            "datalad",
            "get",
            "-n",
            "-r",
            ".",
        ],
        cwd=dataset_root,
    )

def episode_from_path(path: Path) -> str | None:
    match = EPISODE_RE.search(path.stem.lower())
    return match.group(1).lower() if match else None


def discover(root: Path, suffixes: set[str]) -> dict[str, Path]:
    """Discover materialised files and broken annex symlinks."""
    result: dict[str, Path] = {}
    if not root.exists():
        return result

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in suffixes:
            continue
        if not (path.is_file() or path.is_symlink()):
            continue

        episode = episode_from_path(path)
        if episode is None:
            continue

        previous = result.get(episode)
        if previous is not None and previous != path:
            raise RuntimeError(f"Multiple files found for {episode}: {previous} and {path}")
        result[episode] = path

    return result


@dataclass(frozen=True)
class StimulusIndex:
    movies: dict[str, Path]
    transcripts: dict[str, Path]

    @property
    def episodes(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.movies) | set(self.transcripts)))

    def files_for_episode(
        self,
        episode: str,
        *,
        include_movie: bool = True,
        include_transcript: bool = True,
        require_all: bool = True,
    ) -> list[Path]:
        episode = episode.lower()
        paths: list[Path] = []
        missing: list[str] = []

        if include_movie:
            movie = self.movies.get(episode)
            if movie is None:
                missing.append("movie")
            else:
                paths.append(movie)

        if include_transcript:
            transcript = self.transcripts.get(episode)
            if transcript is None:
                missing.append("transcript")
            else:
                paths.append(transcript)

        if missing and require_all:
            raise KeyError(f"{episode}: missing " + ", ".join(missing))

        return paths


def build_index(dataset_root: Path) -> StimulusIndex:
    return StimulusIndex(
        movies=discover(dataset_root / "stimuli" / "movies", {".mkv", ".mp4"}),
        transcripts=discover(dataset_root / "stimuli" / "transcripts", {".tsv"}),
    )


def relative_paths(
    dataset_root: Path,
    paths: Iterable[Path],
) -> list[str]:
    dataset_root = dataset_root.absolute()
    result: list[str] = []

    for path in paths:
        path = path.absolute()
        try:
            rel = path.relative_to(dataset_root)
        except ValueError as exc:
            raise RuntimeError(f"Path is outside dataset root: {path}") from exc
        result.append(str(rel).replace(os.sep, "/"))

    return result


def datalad_get(
    dataset_root: Path,
    paths: Iterable[Path],
    jobs: int = 8,
) -> None:
    paths = list(dict.fromkeys(paths))
    if not paths:
        return
    if jobs < 1:
        raise ValueError("jobs must be positive")

    run(
        [
            "datalad",
            "get",
            f"-J{jobs}",
            "--",
            *relative_paths(dataset_root, paths),
        ],
        cwd=dataset_root,
    )


def datalad_drop(
    dataset_root: Path,
    paths: Iterable[Path],
) -> None:
    """Drop materialised content while preserving dataset metadata."""
    paths = list(dict.fromkeys(paths))
    if not paths:
        return

    run(
        [
            "datalad",
            "drop",
            "--",
            *relative_paths(dataset_root, paths),
        ],
        cwd=dataset_root,
    )


# Compatibility aliases for existing extractors that still import annex_get
# and annex_drop. The implementation is DataLad-native.
_ACTIVE_DATASET_ROOT: Path | None = None


def set_active_dataset_root(dataset_root: Path) -> None:
    global _ACTIVE_DATASET_ROOT
    _ACTIVE_DATASET_ROOT = dataset_root.absolute()


def annex_get(paths: Iterable[Path], jobs: int = 8) -> None:
    if _ACTIVE_DATASET_ROOT is None:
        raise RuntimeError(
            "Dataset root is not configured. "
            "Call set_active_dataset_root(dataset_root) first."
        )
    datalad_get(_ACTIVE_DATASET_ROOT, paths, jobs=jobs)


def annex_drop(paths: Iterable[Path]) -> None:
    if _ACTIVE_DATASET_ROOT is None:
        raise RuntimeError(
            "Dataset root is not configured. "
            "Call set_active_dataset_root(dataset_root) first."
        )
    datalad_drop(_ACTIVE_DATASET_ROOT, paths)

@dataclass(frozen=True)
class Batch:
    index: int
    episodes: tuple[str, ...]


def make_batches(episodes: Iterable[str], batch_size: int) -> list[Batch]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    items = sorted(dict.fromkeys(ep.lower() for ep in episodes))
    return [
        Batch(i, tuple(items[start : start + batch_size]))
        for i, start in enumerate(range(0, len(items), batch_size))
    ]


def batch_paths(
    index: StimulusIndex,
    batch: Batch,
    *,
    include_movie: bool = True,
    include_transcript: bool = True,
    require_all: bool = True,
) -> list[Path]:
    paths: list[Path] = []
    for episode in batch.episodes:
        paths.extend(
            index.files_for_episode(
                episode,
                include_movie=include_movie,
                include_transcript=include_transcript,
                require_all=require_all,
            )
        )
    return list(dict.fromkeys(paths))


def iter_materialized_batches(
    dataset_root: Path,
    *,
    batch_size: int = 1,
    start_batch: int = 0,
    max_batches: int | None = None,
    annex_jobs: int = 8,
    include_movie: bool = True,
    include_transcript: bool = True,
    require_all: bool = True,
    drop_after: bool = True,
) -> Iterator[tuple[Batch, list[Path]]]:
    """Yield DataLad-materialised batches and optionally drop each after use."""
    set_active_dataset_root(dataset_root)
    index = build_index(dataset_root)
    selected_episodes = []
    for episode in index.episodes:
        try:
            files = index.files_for_episode(
                episode,
                include_movie=include_movie,
                include_transcript=include_transcript,
                require_all=require_all,
            )
        except KeyError:
            continue
        if files:
            selected_episodes.append(episode)

    batches = make_batches(selected_episodes, batch_size)
    if start_batch < 0:
        raise ValueError("start_batch must be non-negative")

    end = len(batches) if max_batches is None else min(len(batches), start_batch + max_batches)

    for batch in batches[start_batch:end]:
        paths = batch_paths(
            index,
            batch,
            include_movie=include_movie,
            include_transcript=include_transcript,
            require_all=require_all,
        )
        datalad_get(dataset_root, paths, jobs=annex_jobs)
        try:
            yield batch, paths
        finally:
            if drop_after:
                datalad_drop(dataset_root, paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-branch", default=DEFAULT_DATASET_BRANCH)
    parser.add_argument("--action", choices=("get", "drop", "list"), default="list")
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--annex-jobs", type=int, default=8)
    parser.add_argument("--content", choices=("movies", "transcripts", "both"), default="both")
    parser.add_argument("--allow-missing-pair", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    enable_dataset(args.dataset_root, args.dataset_branch)
    set_active_dataset_root(args.dataset_root)
    index = build_index(args.dataset_root)

    include_movie = args.content in ("movies", "both")
    include_transcript = args.content in ("transcripts", "both")
    require_all = not args.allow_missing_pair

    if args.episode:
        episodes = [ep.lower() for ep in args.episode]
        batches = make_batches(episodes, args.batch_size)
    else:
        eligible: list[str] = []
        for episode in index.episodes:
            try:
                files = index.files_for_episode(
                    episode,
                    include_movie=include_movie,
                    include_transcript=include_transcript,
                    require_all=require_all,
                )
            except KeyError:
                continue
            if files:
                eligible.append(episode)
        batches = make_batches(eligible, args.batch_size)

    end = len(batches) if args.max_batches is None else min(
        len(batches), args.start_batch + args.max_batches
    )

    for batch in batches[args.start_batch:end]:
        paths = batch_paths(
            index,
            batch,
            include_movie=include_movie,
            include_transcript=include_transcript,
            require_all=require_all,
        )
        print(f"batch={batch.index} episodes={','.join(batch.episodes)}")
        for path in paths:
            print(path)

        if args.action == "get":
            datalad_get(args.dataset_root, paths, jobs=args.annex_jobs)
        elif args.action == "drop":
            datalad_drop(args.dataset_root, paths)


if __name__ == "__main__":
    main()
