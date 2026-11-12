"""Train SMRTnet on a Thesis A window bundle using its upstream fold recipe."""

from __future__ import print_function

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from functools import partial
from functools import lru_cache
from pathlib import Path

import dgl
import numpy as np
import torch
from dgllife.utils import (
    CanonicalAtomFeaturizer,
    CanonicalBondFeaturizer,
    smiles_to_bigraph,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from LM_Mol.tokenizer import MolTranBertTokenizer
from loop import train, valid
from model import SmrtNet
from utils import GradualWarmupScheduler
from aggregation import aggregate_window_scores

WINDOW_SIZE = 31
DNA_HIDDEN_SIZE = 1024
BASE_CHANNELS = {"A": 0, "C": 1, "G": 2, "T": 3}
STRUCTURE_CHANNELS = {"(": 0, ")": 1, ".": 2}
PARTITIONS = ("train", "val", "test")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=4)
def _bundle_tables(bundle_path):
    bundle_dir = Path(bundle_path)
    with open(bundle_dir / "bundle_manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("unsupported or incomplete SMRTnet bundle manifest")
    for name, expected in manifest.get("table_sha256", {}).items():
        if _sha256_file(bundle_dir / name) != expected:
            raise ValueError("bundle table fingerprint mismatch for " + name)
    windows = {row["window_id"]: row for row in _read_csv(bundle_dir / "windows.csv")}
    if len(windows) != manifest.get("windows"):
        raise ValueError("bundle window IDs are duplicated or missing")
    pairs = {row["example_id"]: row for row in _read_csv(bundle_dir / "pairs.csv")}
    return manifest, windows, pairs


@lru_cache(maxsize=2)
def _fold_assignments(bundle_path, fold_id):
    bundle_dir = Path(bundle_path)
    by_partition = {partition: [] for partition in PARTITIONS}
    seen = set()
    for assignment in _read_csv(bundle_dir / "window_assignments.csv"):
        if int(assignment["fold_id"]) != int(fold_id):
            continue
        window_id = assignment["window_id"]
        if window_id in seen:
            raise ValueError("duplicate window assignment in selected fold")
        seen.add(window_id)
        if assignment["partition"] not in by_partition:
            raise ValueError("bundle contains an unsupported partition")
        by_partition[assignment["partition"]].append(
            (window_id, int(assignment["label"]))
        )
    return by_partition


def _write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


class SmrtNetWindowDataset(Dataset):
    """Read one partition's pair-window rows and validate referenced frozen states."""

    def __init__(self, bundle_dir, fold_id, partition):
        self.bundle_dir = Path(bundle_dir).resolve()
        if partition not in PARTITIONS:
            raise ValueError("partition must be train, val, or test")
        self.manifest, windows, pairs = _bundle_tables(str(self.bundle_dir))
        if self.manifest.get("schema_version") != 1 or self.manifest.get("status") != "complete":
            raise ValueError("unsupported or incomplete SMRTnet bundle manifest")
        if self.manifest.get("window_size") != WINDOW_SIZE or self.manifest.get("fold_ids") != list(range(5)):
            raise ValueError("bundle window size or five-fold contract is invalid")
        if self.manifest.get("embedding", {}).get("hidden_size") != DNA_HIDDEN_SIZE:
            raise ValueError("bundle GB.DNA hidden size does not match SMRTnet adapter")
        self.pair_by_id = pairs
        self.rows = []
        for window_id, label in _fold_assignments(str(self.bundle_dir), int(fold_id))[partition]:
            if window_id not in windows:
                raise ValueError("window assignment references an unknown window " + window_id)
            row = dict(windows[window_id])
            row["fold_id"] = int(fold_id)
            row["partition"] = partition
            if row["example_id"] not in pairs:
                raise ValueError("window references an unknown parent pair " + row["example_id"])
            if int(row["label"]) != label:
                raise ValueError("window label differs from its copied parent label")
            if len(row["sequence"]) != WINDOW_SIZE or len(row["structure"]) != WINDOW_SIZE:
                raise ValueError("every SMRTnet DNA sequence and structure window must have length 31")
            if set(row["sequence"]) - set(BASE_CHANNELS):
                raise ValueError("DNA windows must use only A/C/G/T")
            if set(row["structure"]) - set(STRUCTURE_CHANNELS):
                raise ValueError("DNA MFE windows contain unsupported dot-bracket symbols")
            self.rows.append(row)
        if not self.rows:
            raise ValueError("selected fold partition has no DNA windows")
        self.embedding_entries = self.manifest.get("embedding_entries", {})
        self._embedding_cache = {}
        self._graph_cache = {}
        self._graph_builder = partial(smiles_to_bigraph, add_self_loop=True)
        self._node_featurizer = CanonicalAtomFeaturizer()
        self._edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

    def __len__(self):
        return len(self.rows)

    def _embedding(self, row):
        key = row["embedding_key"]
        if key not in self._embedding_cache:
            entry = self.embedding_entries.get(key)
            if not isinstance(entry, dict):
                raise ValueError("bundle has no GB.DNA state for window key " + key)
            if entry.get("model_revision") != self.manifest["embedding"]["revision"]:
                raise ValueError("GB.DNA model revision mismatch for window key " + key)
            path = self.bundle_dir / entry["file"]
            if _sha256_file(path) != entry.get("file_sha256"):
                raise ValueError("GB.DNA embedding checksum mismatch for window key " + key)
            states = np.load(path, allow_pickle=False)
            if states.shape != (WINDOW_SIZE, DNA_HIDDEN_SIZE) or not np.isfinite(states).all():
                raise ValueError("GB.DNA states must have finite shape [31, 1024]")
            if _sha256_text(row["sequence"]) != entry.get("sequence_sha256"):
                raise ValueError("GB.DNA state sequence hash does not match the window")
            self._embedding_cache[key] = torch.from_numpy(
                np.asarray(states, dtype=np.float32).copy()
            )
        return self._embedding_cache[key]

    def _molecule_graph(self, row):
        smiles = row["canonical_smiles"]
        if smiles not in self._graph_cache:
            try:
                graph = self._graph_builder(
                    smiles=smiles,
                    node_featurizer=self._node_featurizer,
                    edge_featurizer=self._edge_featurizer,
                )
            except Exception as exc:
                raise ValueError(
                    "SMRTnet could not construct the ligand graph for example "
                    + row["example_id"]
                    + ": "
                    + str(exc)
                )
            self._graph_cache[smiles] = graph
        return self._graph_cache[smiles].clone()

    def __getitem__(self, index):
        row = self.rows[index]
        target = np.zeros((1, WINDOW_SIZE, 5), dtype=np.float32)
        for position, base in enumerate(row["sequence"]):
            target[0, position, BASE_CHANNELS[base]] = 1.0
            target[0, position, 4] = STRUCTURE_CHANNELS[row["structure"][position]]
        return (
            self._molecule_graph(row),
            row["canonical_smiles"],
            target,
            self._embedding(row),
            np.int64(row["label"]),
        )


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dgl_collate_func_dna(items, tokenizer_path="./LM_Mol/bert_vocab.txt"):
    """Batch the ligand graph, structure view, and frozen DNA token states.

    For ``B`` pair windows, the CNN tensor has shape ``[B, 1, 31, 5]``: axis 1 is
    SMRTnet's singleton input plane, axis 2 is the 31-nt position order, and axis 3
    contains A/C/G/T one-hot values followed by the structure category index. Frozen
    GB.DNA states have shape ``[B, 31, 1024]``: ``B`` is batch size and 1024 is the
    model's hidden width.
    """

    graphs, smiles, targets, dna_states, labels = zip(*items)
    graph_batch = dgl.batch(graphs)
    tokenizer = MolTranBertTokenizer(tokenizer_path)
    molecule_tokens = tokenizer.batch_encode_plus(
        ["".join(value) for value in smiles], padding=True, add_special_tokens=True
    )
    target_batch = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
    dna_batch = torch.stack(dna_states, dim=0)
    label_batch = np.asarray(labels)
    return graph_batch, molecule_tokens, target_batch, dna_batch, label_batch


class _Tee(object):
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def write(self, text):
        self.first.write(text)
        self.second.write(text)
        self.first.flush()
        self.second.flush()

    def flush(self):
        self.first.flush()
        self.second.flush()


def _model_args(parsed, fold_dir):
    args = argparse.Namespace()
    args.dna_adaptation = True
    args.mode = "SPU"
    args.seed = parsed.seed
    args.cuda = parsed.cuda
    args.epoch = parsed.epochs
    args.patiences = 1 if parsed.smoke else parsed.patience
    args.batch_size = parsed.batch_size
    args.lr = 1e-4
    args.lr_bert = 1e-4
    args.wd = 1e-6
    args.lm_ft = ["molformer", "rnalm"]
    args.lm_rna_config = ""
    args.lm_rna_model = ""
    args.lm_mol_config = parsed.lm_mol_config
    args.lm_mol_model = parsed.lm_mol_model
    args.hidden_dim_graph = 128
    args.hidden_dim_rna = 128
    args.hidden_dim_rna_lm = 128
    args.hidden_dim_mol_lm = 128
    args.cls = [1024, 1024, 1024, 512]
    args.kernal = 7
    args.pad = 3
    args.channel = 16
    args.gnn_in_feats = 74
    args.gnn_num_layers = 3
    args.gnn_hid_dim_drug = 256
    args.gat_num_heads = 3
    args.gat_feat_drops = 0.2
    args.gat_attn_drops = 0.2
    args.output_dir = str(fold_dir)
    return args


def _smoke_subset(dataset, limit):
    if limit < 2:
        raise ValueError("smoke example limit must be at least two")
    labels = [int(row["label"]) for row in dataset.rows]
    selected = list(range(min(limit, len(labels))))
    if len({labels[index] for index in selected}) < 2 and len(set(labels)) == 2:
        wanted = 1 - labels[selected[0]]
        replacement = next(index for index, label in enumerate(labels) if label == wanted)
        selected[-1] = replacement
    return Subset(dataset, selected)


def _predict_window_rows(model, dataset, tokenizer_path, device, fold_id):
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(dgl_collate_func_dna, tokenizer_path=tokenizer_path),
    )
    model.eval()
    rows = []
    cursor = 0
    with torch.no_grad():
        for graph, molecule_tokens, structures, dna_states, labels in loader:
            graph = graph.to(device)
            molecule_ids = torch.as_tensor(molecule_tokens["input_ids"], device=device)
            molecule_mask = torch.as_tensor(molecule_tokens["attention_mask"], device=device)
            logits = model(
                graph,
                molecule_ids,
                molecule_mask,
                structures.float().to(device),
                dna_embeddings=dna_states.to(device),
            )
            probabilities = torch.sigmoid(logits).reshape(-1).detach().cpu().tolist()
            for probability in probabilities:
                source = dataset.rows[cursor]
                rows.append(
                    {
                        "fold_id": fold_id,
                        "example_id": source["example_id"],
                        "aptamer_id": source["aptamer_id"],
                        "molecule_id": source["molecule_id"],
                        "window_id": source["window_id"],
                        "window_index": int(source["window_index"]),
                        "start_offset": int(source["start_offset"]),
                        "label": int(source["label"]),
                        "binding_probability": float(probability),
                    }
                )
                cursor += 1
    return rows


def _pair_prediction_rows(window_rows, pair_by_id):
    by_example = {}
    for row in window_rows:
        by_example.setdefault(row["example_id"], []).append(row)
    predictions = []
    for example_id, rows in by_example.items():
        rows.sort(key=lambda row: row["window_index"])
        pair = pair_by_id[example_id]
        predictions.append(
            {
                "fold_id": rows[0]["fold_id"],
                "example_id": example_id,
                "aptamer_id": pair["aptamer_id"],
                "molecule_id": pair["molecule_id"],
                "label": int(pair["label"]),
                "binding_probability": aggregate_window_scores(
                    [row["binding_probability"] for row in rows]
                ),
                "window_count": len(rows),
            }
        )
    predictions.sort(key=lambda row: row["example_id"])
    return predictions


def _run_fold(parsed, bundle, run_dir, fold_id, global_class_weight):
    fold_dir = Path(run_dir) / ("fold_" + str(fold_id))
    fold_dir.mkdir(parents=True)
    train_set = SmrtNetWindowDataset(bundle, fold_id, "train")
    valid_set = SmrtNetWindowDataset(bundle, fold_id, "val")
    test_set = SmrtNetWindowDataset(bundle, fold_id, "test")
    if parsed.smoke:
        train_set = _smoke_subset(train_set, parsed.smoke_examples)
        valid_set = _smoke_subset(valid_set, parsed.smoke_examples)
        test_set = _smoke_subset(test_set, parsed.smoke_examples)
    tokenizer_path = parsed.lm_mol_config
    collate = partial(dgl_collate_func_dna, tokenizer_path=tokenizer_path)
    train_loader = DataLoader(
        train_set, batch_size=parsed.batch_size, shuffle=True, num_workers=0, drop_last=False,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=parsed.batch_size, shuffle=True, num_workers=0, drop_last=False,
        collate_fn=collate,
    )
    device = torch.device("cuda:" + str(parsed.cuda))
    torch.cuda.set_device(parsed.cuda)
    args = _model_args(parsed, fold_dir)
    bundle_manifest = train_set.dataset.manifest if isinstance(train_set, Subset) else train_set.manifest
    model = SmrtNet(args).to(device)

    pretrained_ids = set(id(parameter) for parameter in model.tok_emb.parameters())
    pretrained_ids.update(id(parameter) for parameter in model.blocks.parameters())
    downstream_parameters = filter(
        lambda parameter: id(parameter) not in pretrained_ids,
        model.parameters(),
    )
    optimizer = torch.optim.Adam(
        [
            {"params": downstream_parameters, "lr": args.lr},
            {"params": model.tok_emb.parameters(), "lr": args.lr_bert},
            {"params": model.blocks.parameters(), "lr": args.lr_bert},
        ],
        betas=(0.9, 0.999),
        weight_decay=args.wd,
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=8, total_epoch=float(args.epoch), after_scheduler=None
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(global_class_weight), device=device)
    )
    log_path = fold_dir / "train.log"
    best_auc = float("-inf")
    best_epoch = 0
    checkpoint_path = fold_dir / "best_model.pt"
    with open(log_path, "a", encoding="utf-8") as log_handle:
        tee = _Tee(sys.stdout, log_handle)
        with redirect_stdout(tee), redirect_stderr(tee):
            print("Fold", fold_id, "window counts", {
                "train": len(train_set), "val": len(valid_set), "test": len(test_set)
            })
            for epoch in range(1, args.epoch + 1):
                train(args, model, device, train_loader, optimizer, criterion, epoch)
                valid_metrics, valid_loss, _, _ = valid(
                    args, model, device, valid_loader, criterion, epoch
                )
                scheduler.step(epoch)
                valid_auc = float(valid_metrics.auc)
                if valid_auc > best_auc:
                    best_auc = valid_auc
                    best_epoch = epoch
                    torch.save(copy.deepcopy(model.state_dict()), checkpoint_path)
                print(
                    "epoch={0} validation_auc={1:.8f} validation_loss={2:.8f} "
                    "best_epoch={3}".format(epoch, valid_auc, valid_loss, best_epoch)
                )
                if epoch - best_epoch >= args.patiences:
                    print("early stop at epoch", best_epoch)
                    break
    if best_epoch == 0:
        raise RuntimeError("validation ROC-AUC did not select a checkpoint")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restored = SmrtNet(args).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    restored.load_state_dict(state)
    test_dataset = test_set.dataset if isinstance(test_set, Subset) else test_set
    if isinstance(test_set, Subset):
        test_dataset = _IndexedMetadataDataset(test_dataset, test_set.indices)
    window_predictions = _predict_window_rows(
        restored, test_dataset, tokenizer_path, device, fold_id
    )
    pair_predictions = _pair_prediction_rows(window_predictions, test_dataset.pair_by_id)
    _write_csv(
        fold_dir / "window_predictions.csv",
        ("fold_id", "example_id", "aptamer_id", "molecule_id", "window_id", "window_index", "start_offset", "label", "binding_probability"),
        window_predictions,
    )
    _write_csv(
        fold_dir / "pair_predictions.csv",
        ("fold_id", "example_id", "aptamer_id", "molecule_id", "label", "binding_probability", "window_count"),
        pair_predictions,
    )
    config = {
        "fold_id": fold_id,
        "strategy": bundle_manifest["split_strategy"],
        "bundle_schema_version": bundle_manifest["schema_version"],
        "source_fingerprint": bundle_manifest["source_fingerprint"],
        "adapted_fork_commit": bundle_manifest["source_control"]["adapted_fork_commit"],
        "parent_gitlink_sha": bundle_manifest["source_control"]["parent_gitlink_sha"],
        "upstream_base_commit": bundle_manifest["source_control"]["upstream_base_commit"],
        "global_window_class_counts": parsed.global_window_class_counts,
        "inherited_global_positive_class_weight": global_class_weight,
        "training": {
            "seed": parsed.seed,
            "batch_size": parsed.batch_size,
            "epochs": parsed.epochs,
            "patience": parsed.patience,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "scheduler_multiplier": 8,
            "gradient_clip_norm": 5,
            "checkpoint_selection": "validation ROC-AUC",
            "loss": "BCEWithLogitsLoss",
        },
        "selected_epoch": best_epoch,
        "validation_roc_auc": best_auc,
        "rows": {
            "train_windows": len(train_set),
            "validation_windows": len(valid_set),
            "test_windows": len(test_set),
            "test_pairs": len(pair_predictions),
        },
        "smoke": parsed.smoke,
    }
    with open(fold_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(fold_dir / "eval.log", "w", encoding="utf-8") as handle:
        handle.write(
            "Restored selected checkpoint and wrote one window and pair prediction per "
            "held-out test sample.\n"
        )
    return config


class _IndexedMetadataDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        self.rows = [dataset.rows[index] for index in self.indices]
        self.pair_by_id = dataset.pair_by_id
        self.manifest = dataset.manifest

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def run_strategy(parsed):
    bundle_dir = Path(parsed.bundle_dir).resolve()
    with open(bundle_dir / "bundle_manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("split_strategy") != parsed.strategy:
        raise ValueError("--strategy does not match the bundle split strategy")
    if manifest.get("fold_ids") != list(range(5)):
        raise ValueError("SMRTnet requires exactly five shared folds")
    windows = _read_csv(bundle_dir / "windows.csv")
    positive = sum(int(row["label"]) == 1 for row in windows)
    negative = sum(int(row["label"]) == 0 for row in windows)
    if not positive or not negative:
        raise ValueError("global BCE class weight requires both binary classes in windows.csv")
    global_class_weight = int(negative / positive)
    parsed.global_window_class_counts = {"negative": negative, "positive": positive}
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = (
        Path(parsed.output_root).resolve()
        / manifest["dataset_id"]
        / "cross_validation"
        / parsed.strategy
        / run_id
    )
    run_dir.mkdir(parents=True)
    fold_ids = [parsed.fold_id] if parsed.fold_id is not None else list(range(5))
    fold_configs = [
        _run_fold(parsed, bundle_dir, run_dir, fold_id, global_class_weight)
        for fold_id in fold_ids
    ]
    summary = {
        "status": "smoke_check" if parsed.smoke else "folds_complete",
        "smoke": parsed.smoke,
        "run_id": run_id,
        "dataset_id": manifest["dataset_id"],
        "strategy": parsed.strategy,
        "bundle_dir": str(bundle_dir),
        "source_fingerprint": manifest["source_fingerprint"],
        "fold_ids_completed": fold_ids,
        "folds": fold_configs,
        "global_window_class_counts": parsed.global_window_class_counts,
        "inherited_global_positive_class_weight": global_class_weight,
        "excluded_rows": manifest["excluded_rows"],
        "benchmark_metrics_complete": False,
        "note": "Use the Thesis A comparison CLI after five-fold predictions are complete.",
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("SMRTnet run artifacts:", run_dir)
    return run_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--fold-id", type=int, choices=range(5))
    parser.add_argument("--output-root", default="../../outputs/sota_comparison/smrtnet")
    parser.add_argument("--lm-mol-config", default="./LM_Mol/bert_vocab.txt")
    parser.add_argument(
        "--lm-mol-model",
        default="./LM_Mol/pretrained/checkpoints/N-Step-Checkpoint_3_30000.ckpt",
    )
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-examples", type=int, default=4)
    return parser


def main(argv=None):
    parsed = build_parser().parse_args(argv)
    if parsed.fold_id is None and parsed.smoke:
        raise SystemExit("--smoke requires --fold-id")
    if parsed.epochs < 1 or parsed.batch_size < 1 or parsed.patience < 1:
        raise SystemExit("epochs, batch size, and patience must be positive")
    if parsed.smoke:
        parsed.epochs = 1
    if not torch.cuda.is_available():
        raise SystemExit("SMRTnet requires its CUDA/DGL environment; no CUDA device is available")
    if parsed.cuda < 0 or parsed.cuda >= torch.cuda.device_count():
        raise SystemExit("--cuda refers to an unavailable logical GPU")
    run_strategy(parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
