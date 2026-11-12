"""Command-line entry point for preparing DNA-adapted SMRTnet input bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .thesis_a_bundle import prepare_bundle
except ImportError:  # Running this file directly from the Thesis A project root.
    import sys

    _SMRTNET_ROOT = Path(__file__).resolve().parent
    if str(_SMRTNET_ROOT) not in sys.path:
        sys.path.insert(0, str(_SMRTNET_ROOT))
    from thesis_a_bundle import prepare_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device {device!r} requested but CUDA is unavailable")
    manifest = prepare_bundle(
        args.split_dir,
        args.output_dir,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        device=device,
        overwrite=args.overwrite,
    )
    print(
        f"Prepared {manifest['windows']} windows from {manifest['eligible_pairs']} pairs "
        f"({manifest['unique_window_sequences']} unique DNA windows) at {args.output_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI invocation
    raise SystemExit(main())
