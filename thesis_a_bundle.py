"""Build a provenance-checked SMRTnet input bundle from shared Thesis A CV rows.

Bundle DNA windows have shape ``[31]`` and frozen GB.DNA states have shape ``[31, 1024]``;
31 is the upstream sequence-window width and 1024 is the pinned GB.DNA hidden size. Fold
assignments are stored separately from window records so each parent window is counted once
when reproducing SMRTnet's inherited global class-weight calculation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

_THESIS_ROOT = Path(__file__).resolve().parents[2]
_THESIS_SRC = _THESIS_ROOT / "src"
if str(_THESIS_SRC) not in sys.path:
    sys.path.insert(0, str(_THESIS_SRC))

from data_featurization.split_reader import read_split  # noqa: E402
from dataset_preprocessing.io import sha256_file  # noqa: E402
from dna_feature_encoding.batching.tokenizer import (  # noqa: E402
    GBDNATokenizerAdapter,
    GB_DNA_REPOSITORY,
    GB_DNA_REVISION,
    load_gbdna_embed_task,
)
from dna_feature_encoding.config import EncodingConfig  # noqa: E402
from dna_feature_encoding.structure.vienna import _load_dna_parameters  # noqa: E402

BUNDLE_SCHEMA_VERSION = 1
WINDOW_SIZE = 31
WINDOW_STRIDE = 1
GB_DNA_HIDDEN_SIZE = 1024
GB_DNA_MODEL = "GB.DNA-300M"
UPSTREAM_URL = "https://github.com/Yuhan-Fei/SMRTnet.git"
FORK_URL = "https://github.com/Vincenzoos/SMRTnet.git"
UPSTREAM_BASE_COMMIT = "3bb10428207cafcefa860defdd8e72cd5ea47988"
FORK_SUBMODULE_PATH = "thesisA-proposal/benchmarks/smrtnet"
_PARTITIONS = ("train", "val", "test")


class SmrtNetBundleError(RuntimeError):
    """Shared rows or generated bundle inputs violate the SMRTnet contract."""


@dataclass(frozen=True)
class AptamerStructure:
    """One full-length MFE structure aligned to a canonical aptamer sequence."""

    aptamer_id: str
    sequence: str
    structure: str
    mfe_energy: float
    sequence_sha256: str


@dataclass(frozen=True)
class WindowRecord:
    """One overlapping, label-copied DNA sequence/structure model input."""

    window_id: str
    example_id: str
    aptamer_id: str
    molecule_id: str
    canonical_smiles: str
    label: int
    window_index: int
    start_offset: int
    sequence: str
    structure: str
    sequence_sha256: str
    embedding_key: str


class ViennaDNAMFEAdapter:
    """Compute one MFE dot-bracket structure per full DNA sequence.

    MFE inputs use shape ``[L]``, where ``L`` is the full canonical aptamer length. The
    returned dot-bracket string also has shape ``[L]`` and is sliced only after folding.
    """

    def __init__(
        self,
        config: EncodingConfig | None = None,
        *,
        rna_module: Any | None = None,
    ) -> None:
        self.config = config or EncodingConfig()
        self._rna = rna_module

    @property
    def rna(self) -> Any:
        if self._rna is None:
            try:
                import RNA
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise SmrtNetBundleError(
                    "ViennaRNA Python bindings are required to prepare DNA MFE structures"
                ) from exc
            self._rna = RNA
        return self._rna

    @property
    def version(self) -> str:
        value = getattr(self.rna, "__version__", None)
        return str(value) if value is not None else "unknown"

    def compute(self, sequence: str) -> tuple[str, float]:
        try:
            rna = self.rna
            _load_dna_parameters(rna, self.config.vienna_parameter_set)
            model_details = rna.md()
            model_details.temperature = self.config.vienna_temperature_celsius
            structure, energy = rna.fold_compound(sequence, model_details).mfe()
        except Exception as exc:
            if isinstance(exc, SmrtNetBundleError):
                raise
            raise SmrtNetBundleError(
                f"ViennaRNA DNA MFE prediction failed for a sequence of length "
                f"{len(sequence)}: {exc}"
            ) from exc
        structure = str(structure)
        if len(structure) != len(sequence) or set(structure) - {".", "(", ")"}:
            raise SmrtNetBundleError(
                f"ViennaRNA returned a dot-bracket string not aligned to length {len(sequence)}"
            )
        energy_value = float(energy)
        if not np.isfinite(energy_value):
            raise SmrtNetBundleError("ViennaRNA returned a non-finite MFE energy")
        return structure, energy_value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _embedding_key(sequence: str, revision: str = GB_DNA_REVISION) -> str:
    sequence_hash = _sha256_text(sequence)
    return hashlib.sha256(f"{revision}:{sequence_hash}".encode("ascii")).hexdigest()


def _rdkit_validate_smiles(smiles: str, example_id: str) -> None:
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise SmrtNetBundleError("RDKit is required to validate canonical SMILES") from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise SmrtNetBundleError(f"example {example_id!r} has invalid canonical SMILES {smiles!r}")
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    if canonical != smiles:
        raise SmrtNetBundleError(
            f"example {example_id!r} SMILES is not canonical: {smiles!r} (expected {canonical!r})"
        )


def read_shared_cv_split(split_dir: str | Path) -> Any:
    """Read and validate a shared five-fold processed split without feature artifacts."""

    try:
        split = read_split(split_dir)
    except Exception as exc:
        raise SmrtNetBundleError(f"could not resolve shared split {split_dir}: {exc}") from exc
    if split.split_mode != "cross_validation" or split.fold_ids != (0, 1, 2, 3, 4):
        raise SmrtNetBundleError(
            "SMRTnet preparation requires a shared five-fold cross-validation split; "
            f"got mode={split.split_mode!r}, folds={split.fold_ids}"
        )
    for row in split.rows:
        if not row.sequence or set(row.sequence) - {"A", "C", "G", "T"}:
            raise SmrtNetBundleError(
                f"example {row.example_id!r} does not use the canonical A/C/G/T alphabet"
            )
        if row.label not in (0, 1, None):
            raise SmrtNetBundleError(f"example {row.example_id!r} has invalid binary label {row.label!r}")
        _rdkit_validate_smiles(row.canonical_smiles, row.example_id)
        if len(row.memberships) != 5 or {item.fold_id for item in row.memberships} != set(range(5)):
            raise SmrtNetBundleError(
                f"example {row.example_id!r} must have exactly one membership in each fold"
            )
        for membership in row.memberships:
            if membership.partition not in _PARTITIONS:
                raise SmrtNetBundleError(
                    f"example {row.example_id!r} has unsupported partition "
                    f"{membership.partition!r}"
                )
    return split


def _exclusion_reasons(row: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    if row.label is None:
        reasons.append("missing_binary_label")
    if len(row.sequence) < WINDOW_SIZE:
        reasons.append("shorter_than_31_nt")
    return tuple(reasons)


def expand_pair_windows(
    split: Any,
    *,
    structure_adapter: ViennaDNAMFEAdapter,
) -> tuple[tuple[AptamerStructure, ...], tuple[WindowRecord, ...], tuple[dict[str, Any], ...]]:
    """Fold full aptamers, then create aligned 31-nt windows for labeled eligible pairs."""

    structures_by_id: dict[str, AptamerStructure] = {}
    windows: list[WindowRecord] = []
    exclusions: list[dict[str, Any]] = []
    for row in split.rows:
        reasons = _exclusion_reasons(row)
        if reasons:
            exclusions.append(
                {
                    "example_id": row.example_id,
                    "aptamer_id": row.aptamer_id,
                    "molecule_id": row.molecule_id,
                    "sequence_length": len(row.sequence),
                    "reasons": ";".join(reasons),
                }
            )
            continue

        structure_record = structures_by_id.get(row.aptamer_id)
        if structure_record is None:
            structure, energy = structure_adapter.compute(row.sequence)
            structure_record = AptamerStructure(
                aptamer_id=row.aptamer_id,
                sequence=row.sequence,
                structure=structure,
                mfe_energy=energy,
                sequence_sha256=_sha256_text(row.sequence),
            )
            structures_by_id[row.aptamer_id] = structure_record
        elif structure_record.sequence != row.sequence:
            raise SmrtNetBundleError(
                f"aptamer ID {row.aptamer_id!r} maps to different DNA sequences"
            )

        sequence = row.sequence
        full_structure = structure_record.structure
        if len(sequence) != len(full_structure):
            raise SmrtNetBundleError(
                f"example {row.example_id!r} sequence and MFE structure lengths differ"
            )
        window_count = len(sequence) - WINDOW_SIZE + 1
        for window_index in range(window_count):
            start = window_index * WINDOW_STRIDE
            end = start + WINDOW_SIZE
            window_sequence = sequence[start:end]
            window_structure = full_structure[start:end]
            if len(window_sequence) != WINDOW_SIZE or len(window_structure) != WINDOW_SIZE:
                raise SmrtNetBundleError(
                    f"example {row.example_id!r} produced an incomplete window at offset {start}"
                )
            sequence_hash = _sha256_text(window_sequence)
            window_id = "window_" + _sha256_text(f"{row.example_id}:{start}")
            windows.append(
                WindowRecord(
                    window_id=window_id,
                    example_id=row.example_id,
                    aptamer_id=row.aptamer_id,
                    molecule_id=row.molecule_id,
                    canonical_smiles=row.canonical_smiles,
                    label=int(row.label),
                    window_index=window_index,
                    start_offset=start,
                    sequence=window_sequence,
                    structure=window_structure,
                    sequence_sha256=sequence_hash,
                    embedding_key=_embedding_key(window_sequence),
                )
            )
    return tuple(structures_by_id.values()), tuple(windows), tuple(exclusions)


def _hidden_states(output: Any) -> Tensor:
    """Return GB.DNA output shaped ``[B, T, H]``.

    ``B`` is the number of unique DNA windows in the current batch, ``T`` is the tokenizer
    padded token width, and ``H`` is the pinned 1024-wide GB.DNA hidden representation.
    """

    if isinstance(output, Mapping):
        hidden = output.get("last_hidden_state")
    else:
        hidden = getattr(output, "last_hidden_state", output)
    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
        raise SmrtNetBundleError("GB.DNA must return last_hidden_state with shape [B, T, H]")
    return hidden


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_value(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SmrtNetBundleError(f"could not record Git provenance: {exc}") from exc
    return result.stdout.strip()


def _materialize_embeddings(
    windows: tuple[WindowRecord, ...],
    stage_dir: Path,
    *,
    batch_size: int,
    device: torch.device,
    embed_task: Any | None,
    tokenizer_adapter: Any | None,
) -> dict[str, dict[str, Any]]:
    unique_sequences: dict[str, str] = {}
    for window in windows:
        unique_sequences.setdefault(window.embedding_key, window.sequence)
    entries: dict[str, dict[str, Any]] = {}
    if not unique_sequences:
        return entries

    if embed_task is None:
        embed_task = load_gbdna_embed_task()
    if tokenizer_adapter is None:
        tokenizer_adapter = GBDNATokenizerAdapter(embed_task=embed_task)
    if not hasattr(embed_task, "backbone"):
        raise SmrtNetBundleError("configured GB.DNA task does not expose a backbone")
    backbone = embed_task.backbone.to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    keyed_sequences = list(unique_sequences.items())
    embeddings_dir = stage_dir / "embeddings"
    for offset in tqdm(range(0, len(keyed_sequences), batch_size), desc="GB.DNA windows"):
        batch = keyed_sequences[offset : offset + batch_size]
        keys = [key for key, _ in batch]
        sequences = [sequence for _, sequence in batch]
        token_batch = tokenizer_adapter.tokenize(
            sequences,
            sequence_lengths=[WINDOW_SIZE] * len(sequences),
            aptamer_ids=keys,
        )
        input_ids = token_batch.input_ids.to(device)
        attention = token_batch.attention_mask.to(device=device, dtype=torch.bool)
        special = token_batch.special_tokens_mask.to(device=device, dtype=torch.bool)
        with torch.inference_mode():
            output = backbone(input_ids=input_ids, attention_mask=attention)
        hidden = _hidden_states(output)
        if hidden.shape[:2] != input_ids.shape or hidden.shape[2] != GB_DNA_HIDDEN_SIZE:
            raise SmrtNetBundleError(
                f"GB.DNA hidden-state shape must be [B, T, {GB_DNA_HIDDEN_SIZE}], "
                f"got {tuple(hidden.shape)}"
            )
        retained = attention & ~special
        for index, (key, sequence) in enumerate(batch):
            values = hidden[index][retained[index]].detach().cpu().to(torch.float32).numpy()
            if values.shape != (WINDOW_SIZE, GB_DNA_HIDDEN_SIZE):
                raise SmrtNetBundleError(
                    f"GB.DNA token alignment failed for window key {key!r}: {values.shape}"
                )
            if not np.isfinite(values).all():
                raise SmrtNetBundleError(f"GB.DNA returned non-finite states for window {key!r}")
            relative_path = Path("embeddings") / f"{key}.npy"
            _atomic_numpy(stage_dir / relative_path, values)
            entries[key] = {
                "file": relative_path.as_posix(),
                "sequence_sha256": _sha256_text(sequence),
                "model_repository": GB_DNA_REPOSITORY,
                "model_revision": GB_DNA_REVISION,
                "shape": [WINDOW_SIZE, GB_DNA_HIDDEN_SIZE],
                "dtype": "float32",
                "file_sha256": sha256_file(stage_dir / relative_path),
            }
    return entries


def prepare_bundle(
    split_dir: str | Path,
    output_dir: str | Path,
    *,
    dataset_id: str,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
    structure_adapter: ViennaDNAMFEAdapter | None = None,
    embed_task: Any | None = None,
    tokenizer_adapter: Any | None = None,
) -> dict[str, Any]:
    """Create a complete bundle containing rows, assignments, MFE windows, and DNA states."""

    if not dataset_id or Path(dataset_id).name != dataset_id or dataset_id in {".", ".."}:
        raise ValueError("dataset_id must be one path component")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise SmrtNetBundleError(
            f"output path already exists: {destination}; pass --overwrite to replace it"
        )

    split = read_shared_cv_split(split_dir)
    structure_adapter = structure_adapter or ViennaDNAMFEAdapter()
    structures, windows, exclusions = expand_pair_windows(
        split, structure_adapter=structure_adapter
    )
    selected_device = torch.device(device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        rows_by_id = {row.example_id: row for row in split.rows}
        eligible_rows = [
            row
            for row in split.rows
            if not _exclusion_reasons(row)
        ]
        pair_rows = [
            {
                "example_id": row.example_id,
                "aptamer_id": row.aptamer_id,
                "molecule_id": row.molecule_id,
                "sequence": row.sequence,
                "canonical_smiles": row.canonical_smiles,
                "label": row.label,
                "sequence_length": len(row.sequence),
            }
            for row in eligible_rows
        ]
        pair_assignments = [
            {
                "example_id": row.example_id,
                "fold_id": membership.fold_id,
                "partition": membership.partition,
            }
            for row in eligible_rows
            for membership in row.memberships
        ]
        window_rows = [
            {
                "window_id": window.window_id,
                "example_id": window.example_id,
                "aptamer_id": window.aptamer_id,
                "molecule_id": window.molecule_id,
                "canonical_smiles": window.canonical_smiles,
                "label": window.label,
                "window_index": window.window_index,
                "start_offset": window.start_offset,
                "sequence": window.sequence,
                "structure": window.structure,
                "sequence_sha256": window.sequence_sha256,
                "embedding_key": window.embedding_key,
            }
            for window in windows
        ]
        windows_by_example: dict[str, list[WindowRecord]] = {}
        for window in windows:
            windows_by_example.setdefault(window.example_id, []).append(window)
        window_assignments = [
            {
                "window_id": window.window_id,
                "example_id": window.example_id,
                "fold_id": membership.fold_id,
                "partition": membership.partition,
                "label": window.label,
            }
            for example_id, pair_row in rows_by_id.items()
            if example_id in windows_by_example
            for window in windows_by_example[example_id]
            for membership in pair_row.memberships
        ]

        structure_rows = [
            {
                "aptamer_id": item.aptamer_id,
                "sequence_sha256": item.sequence_sha256,
                "sequence_length": len(item.sequence),
                "sequence": item.sequence,
                "structure": item.structure,
                "mfe_energy_kcal_per_mol": item.mfe_energy,
            }
            for item in structures
        ]

        embeddings = _materialize_embeddings(
            windows,
            stage_dir,
            batch_size=batch_size,
            device=selected_device,
            embed_task=embed_task,
            tokenizer_adapter=tokenizer_adapter,
        )
        for path, headers, rows in (
            ("pairs.csv", ("example_id", "aptamer_id", "molecule_id", "sequence", "canonical_smiles", "label", "sequence_length"), pair_rows),
            ("pair_assignments.csv", ("example_id", "fold_id", "partition"), pair_assignments),
            ("windows.csv", ("window_id", "example_id", "aptamer_id", "molecule_id", "canonical_smiles", "label", "window_index", "start_offset", "sequence", "structure", "sequence_sha256", "embedding_key"), window_rows),
            ("window_assignments.csv", ("window_id", "example_id", "fold_id", "partition", "label"), window_assignments),
            ("structures.csv", ("aptamer_id", "sequence_sha256", "sequence_length", "sequence", "structure", "mfe_energy_kcal_per_mol"), structure_rows),
            ("excluded_rows.csv", ("example_id", "aptamer_id", "molecule_id", "sequence_length", "reasons"), exclusions),
        ):
            _atomic_csv(stage_dir / path, headers, rows)

        table_hashes = {
            name: sha256_file(stage_dir / name)
            for name in (
                "pairs.csv",
                "pair_assignments.csv",
                "windows.csv",
                "window_assignments.csv",
                "structures.csv",
                "excluded_rows.csv",
            )
        }
        submodule_head = _git_value(
            _THESIS_ROOT, "-C", str(_THESIS_ROOT / "benchmarks" / "smrtnet"), "rev-parse", "HEAD"
        )
        indexed_gitlink = _git_value(
            _THESIS_ROOT, "rev-parse", f":{FORK_SUBMODULE_PATH}"
        )
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "status": "complete",
            "dataset_id": dataset_id,
            "split_mode": split.split_mode,
            "split_strategy": split.split_strategy,
            "split_dir": Path(split_dir).expanduser().resolve().as_posix(),
            "fold_ids": list(split.fold_ids),
            "source_fingerprint": split.source_fingerprint,
            "canonical_clean_sha256": sha256_file(split.processed_root / "clean.csv"),
            "source_rows": len(split.rows),
            "eligible_pairs": len(pair_rows),
            "excluded_rows": len(exclusions),
            "windows": len(windows),
            "window_assignments": len(window_assignments),
            "unique_window_sequences": len(embeddings),
            "window_size": WINDOW_SIZE,
            "window_stride": WINDOW_STRIDE,
            "structure": {
                "algorithm": "MFE",
                "viennarna_version": structure_adapter.version,
                "parameter_set": structure_adapter.config.vienna_parameter_set,
                "temperature_celsius": structure_adapter.config.vienna_temperature_celsius,
                "alignment": "full-length DNA sequence before 31-nt slicing",
            },
            "embedding": {
                "model": GB_DNA_MODEL,
                "repository": GB_DNA_REPOSITORY,
                "revision": GB_DNA_REVISION,
                "hidden_size": GB_DNA_HIDDEN_SIZE,
                "cache_key": "sha256(model_revision + ':' + sha256(window_sequence))",
            },
            "source_control": {
                "upstream_url": UPSTREAM_URL,
                "upstream_base_commit": UPSTREAM_BASE_COMMIT,
                "fork_url": FORK_URL,
                "adapted_fork_commit": submodule_head,
                "parent_gitlink_sha": indexed_gitlink,
            },
            "table_sha256": table_hashes,
            "embedding_entries": embeddings,
        }
        manifest_path = stage_dir / "bundle_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        backup_dir: Path | None = None
        if destination.exists():
            backup_dir = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
            destination.rename(backup_dir)
        try:
            stage_dir.rename(destination)
        except Exception:
            if backup_dir is not None and backup_dir.exists():
                backup_dir.rename(destination)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
        return manifest
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)


__all__ = [
    "AptamerStructure",
    "BUNDLE_SCHEMA_VERSION",
    "GB_DNA_HIDDEN_SIZE",
    "SmrtNetBundleError",
    "UPSTREAM_BASE_COMMIT",
    "ViennaDNAMFEAdapter",
    "WINDOW_SIZE",
    "WindowRecord",
    "expand_pair_windows",
    "prepare_bundle",
    "read_shared_cv_split",
]
