# Auto-exported from QATACOV19_QATALVIT_NOBERT_050PCT_GPTPRO_FIXED.ipynb
# Keep this script paired with the notebook in the same package.

# %% [code] cell 2
"""Python-compatible export of the paired Kaggle notebook.

Notebook shell commands and writefile cells were converted to ordinary
Python helpers so this file can be checked with `python -m py_compile`.
"""

def _run_notebook_shell(command: str) -> None:
    """Run a shell command that originally appeared as a notebook `!` command."""
    import subprocess
    import sys

    if command.startswith("pip "):
        command = f'"{sys.executable}" -m {command}'
    subprocess.check_call(command, shell=True)


def _write_notebook_file(path: str, content: str) -> None:
    """Write a file that originally came from a notebook `%%writefile` cell."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")

_run_notebook_shell('pip -q install openpyxl transformers sentencepiece protobuf safetensors')

# %% [code] cell 3
_write_notebook_file('/kaggle/working/text_encoder.py', 'import re\nimport threading\nfrom pathlib import Path\nfrom typing import Any, Iterable, List, Sequence\n\nimport torch\n\n\nFEATURE_CACHE_FORMAT_VERSION = 3\nREPORT_UNIT_SPLIT_VERSION = 2\n\nATTRIBUTE_GROUPS = {\n    "laterality": ["unknown", "left", "right", "bilateral", "diffuse"],\n    "vertical": ["unknown", "upper", "middle", "lower", "basal"],\n    "count": ["unknown", "single", "multiple", "diffuse"],\n    "extent": ["unknown", "focal", "multifocal", "diffuse"],\n    "severity": ["unknown", "mild", "moderate", "severe"],\n}\n\n\ndef attribute_vector_size() -> int:\n    return sum(len(options) for options in ATTRIBUTE_GROUPS.values())\n\n\ndef build_cache_metadata(\n    model_name: str,\n    max_units: int,\n    parser_name: str = "structured-report-parser-v1",\n) -> dict[str, Any]:\n    return {\n        "format_version": FEATURE_CACHE_FORMAT_VERSION,\n        "model_name": model_name,\n        "max_units": int(max_units),\n        "parser_name": parser_name,\n        "report_unit_split_version": REPORT_UNIT_SPLIT_VERSION,\n    }\n\n\ndef split_report_into_units(report: str) -> List[str]:\n    text = (report or "").strip()\n    if not text:\n        return ["[NO_TEXT]"]\n\n    raw_units = re.split(r"[\\n.;]+|,\\s+(?=[A-Za-z])", text)\n    units = [unit.strip() for unit in raw_units if unit and unit.strip()]\n    if not units:\n        return ["[NO_TEXT]"]\n    return units\n\n\ndef feature_cache_path(cache_dir: str | Path, sample_id: str) -> Path:\n    stem = Path(sample_id).stem\n    return Path(cache_dir) / f"{stem}.pt"\n\n\ndef save_report_features(\n    cache_dir: str | Path,\n    sample_id: str,\n    text: torch.Tensor,\n    attributes: torch.Tensor,\n    metadata: dict[str, Any] | None = None,\n) -> Path:\n    cache_path = feature_cache_path(cache_dir, sample_id)\n    cache_path.parent.mkdir(parents=True, exist_ok=True)\n    payload = {\n        "text": text.cpu(),\n        "attributes": attributes.cpu(),\n        "metadata": metadata or {"format_version": FEATURE_CACHE_FORMAT_VERSION},\n    }\n    torch.save(payload, cache_path)\n    return cache_path\n\n\ndef _metadata_matches(\n    cached_metadata: dict[str, Any] | None,\n    expected_metadata: dict[str, Any] | None,\n) -> bool:\n    if expected_metadata is None:\n        return True\n    if cached_metadata is None:\n        return False\n    for key, expected_value in expected_metadata.items():\n        if cached_metadata.get(key) != expected_value:\n            return False\n    return True\n\n\ndef load_report_features(\n    cache_dir: str | Path,\n    sample_id: str,\n    expected_metadata: dict[str, Any] | None = None,\n) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None:\n    cache_path = feature_cache_path(cache_dir, sample_id)\n    if not cache_path.exists():\n        return None\n    payload = torch.load(cache_path, map_location="cpu", weights_only=True)\n    metadata = payload.get("metadata", {})\n    if not _metadata_matches(metadata, expected_metadata):\n        return None\n    return payload["text"], payload["attributes"], metadata\n\n\nclass StructuredReportParser:\n    """Heuristic parser that extracts coarse structured report attributes."""\n\n    def __init__(self) -> None:\n        self.parser_name = "structured-report-parser-v1"\n        self.patterns = {\n            "laterality": {\n                "bilateral": re.compile(r"\\bbilateral\\b|\\bboth lungs?\\b", re.IGNORECASE),\n                "left": re.compile(r"\\bleft\\b", re.IGNORECASE),\n                "right": re.compile(r"\\bright\\b", re.IGNORECASE),\n                "diffuse": re.compile(r"\\bdiffuse\\b|\\bscattered\\b", re.IGNORECASE),\n            },\n            "vertical": {\n                "upper": re.compile(r"\\bupper\\b|\\bapical\\b", re.IGNORECASE),\n                "middle": re.compile(r"\\bmiddle\\b|\\bmid\\b", re.IGNORECASE),\n                "lower": re.compile(r"\\blower\\b|\\bbasal\\b|\\bbase\\b", re.IGNORECASE),\n                "basal": re.compile(r"\\bbasal\\b|\\bbase\\b", re.IGNORECASE),\n            },\n            "count": {\n                "single": re.compile(r"\\bsingle\\b|\\bone lesion\\b", re.IGNORECASE),\n                "multiple": re.compile(r"\\bmultiple\\b|\\bseveral\\b|\\bmultifocal\\b", re.IGNORECASE),\n                "diffuse": re.compile(r"\\bdiffuse\\b|\\bwidespread\\b", re.IGNORECASE),\n            },\n            "extent": {\n                "focal": re.compile(r"\\bfocal\\b|\\blocalized\\b", re.IGNORECASE),\n                "multifocal": re.compile(r"\\bmultifocal\\b|\\bpatchy\\b", re.IGNORECASE),\n                "diffuse": re.compile(r"\\bdiffuse\\b|\\bextensive\\b|\\bconfluent\\b", re.IGNORECASE),\n            },\n            "severity": {\n                "mild": re.compile(r"\\bmild\\b", re.IGNORECASE),\n                "moderate": re.compile(r"\\bmoderate\\b", re.IGNORECASE),\n                "severe": re.compile(r"\\bsevere\\b|\\bcritical\\b|\\bextensive\\b", re.IGNORECASE),\n            },\n        }\n\n    def parse(self, report: str) -> dict[str, str]:\n        text = report or ""\n        parsed: dict[str, str] = {}\n        for group, options in ATTRIBUTE_GROUPS.items():\n            parsed[group] = "unknown"\n            for option in options:\n                if option == "unknown":\n                    continue\n                pattern = self.patterns[group].get(option)\n                if pattern is not None and pattern.search(text):\n                    parsed[group] = option\n                    break\n        return parsed\n\n    def vectorize(self, report: str) -> torch.Tensor:\n        parsed = self.parse(report)\n        vector = torch.zeros(attribute_vector_size(), dtype=torch.float32)\n        offset = 0\n        for group, options in ATTRIBUTE_GROUPS.items():\n            value = parsed[group]\n            index = options.index(value)\n            vector[offset + index] = 1.0\n            offset += len(options)\n        return vector\n\n\nclass CachedDomainTextEncoder:\n    """Cached domain-aware text encoder for medical report text."""\n\n    def __init__(\n        self,\n        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",\n        device: str | None = None,\n        local_files_only: bool = False,\n    ) -> None:\n        self.model_name = model_name\n        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")\n        self.local_files_only = local_files_only\n        self._tokenizer = None\n        self._model = None\n        self._cache: dict[str, torch.Tensor] = {}\n        self._lock = threading.Lock()\n        self.report_parser = StructuredReportParser()\n\n    def cache_metadata(self, max_units: int) -> dict[str, Any]:\n        return build_cache_metadata(\n            model_name=self.model_name,\n            max_units=max_units,\n            parser_name=self.report_parser.parser_name,\n        )\n\n    def _ensure_model(self) -> None:\n        if self._model is not None and self._tokenizer is not None:\n            return\n        with self._lock:\n            if self._model is None or self._tokenizer is None:\n                from transformers import AutoModel, AutoTokenizer\n\n                try:\n                    self._tokenizer = AutoTokenizer.from_pretrained(\n                        self.model_name,\n                        local_files_only=self.local_files_only,\n                        use_fast=False,\n                    )\n                    self._model = AutoModel.from_pretrained(\n                        self.model_name,\n                        local_files_only=self.local_files_only,\n                        use_safetensors=True,\n                    )\n                except Exception as exc:\n                    raise RuntimeError(\n                        "Failed to initialize the Hugging Face text encoder. "\n                        "Make sure the requested model is available, the environment has "\n                        "compatible tokenizer dependencies, and offline mode has a cached copy. "\n                        "Recommended setup: transformers<5, protobuf, and sentencepiece installed."\n                    ) from exc\n                self._model.eval()\n                self._model.to(self.device)\n\n    def encode_lines(self, lines: Iterable[str], max_lines: int | None = None) -> torch.Tensor:\n        cleaned_lines: List[str] = [line.strip() for line in lines if line and line.strip()]\n        if max_lines is not None:\n            cleaned_lines = cleaned_lines[:max_lines]\n        if not cleaned_lines:\n            cleaned_lines = ["[NO_TEXT]"]\n        outputs = [self._encode_single(line) for line in cleaned_lines]\n        if max_lines is not None and len(outputs) < max_lines:\n            pad_token = outputs[0].new_zeros(outputs[0].shape)\n            outputs.extend([pad_token.clone() for _ in range(max_lines - len(outputs))])\n        return torch.stack(outputs, dim=0)\n\n    def encode_report(self, report: str, max_lines: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:\n        lines = split_report_into_units(report)\n        tokens = self.encode_lines(lines, max_lines=max_lines)\n        attrs = self.report_parser.vectorize(report)\n        return tokens, attrs\n\n    def batch_attribute_vectors(self, reports: Sequence[str]) -> torch.Tensor:\n        return torch.stack([self.report_parser.vectorize(report) for report in reports], dim=0)\n\n    def _encode_single(self, text: str) -> torch.Tensor:\n        cached = self._cache.get(text)\n        if cached is not None:\n            return cached.clone()\n\n        self._ensure_model()\n        assert self._tokenizer is not None\n        assert self._model is not None\n        with torch.inference_mode():\n            encoded = self._tokenizer(\n                text,\n                truncation=True,\n                max_length=64,\n                padding="max_length",\n                return_tensors="pt",\n            )\n            encoded = {key: value.to(self.device) for key, value in encoded.items()}\n            outputs = self._model(**encoded)\n            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).detach().cpu()\n\n        self._cache[text] = embedding\n        return embedding.clone()\n\n\n# Backward-compatible alias for earlier local changes.\nCachedTextEncoder = CachedDomainTextEncoder\n\n')
# %% [code] cell 4
_write_notebook_file('/kaggle/working/qata_lvit_nobert_gptpro_fixed.py', '# -*- coding: utf-8 -*-\n"""\nCleaned source exported from the original notebook for submission.\nMarkdown cells are preserved as comments so the training and model flow\nremain easy to read for the grader.\n"""\n\n')
# %% markdown cell 0
# # QaTa-Covid19 QaTaLViT no-BERT GPTPro Fixed (Kaggle-ready)
#
# This file keeps the GPTPro-fixed QaTa recipe but uses the no-BERT structured
# prompt pipeline from the optimized QaTaLViT final branch. The short notes
# below keep the main design ideas:
#
# - prompts are normalized by template to reduce text noise,
# - left/right and upper/lower semantics are preserved by avoiding flip and rot90,
# - language features are fused at multiple encoder scales,
# - the loss combines Dice, BCE, Focal Tversky, and image-text alignment,
# - the training pipeline supports mixed precision, warmup, and cosine scheduling.

# %% code cell 1
import os
import re
import math
import copy
import json
import random
import zipfile
import contextlib
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    def display(x):
        """Execute a documented step of the segmentation training pipeline."""
        print(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# -----------------------------
# Global config
# -----------------------------
@dataclass
class CFG:
    """Implementation class used by the segmentation training pipeline."""
    seed: int = 42
    image_size: int = 224
    epochs: int = 50
    model_name: str = "QaTaLViT-NoBERT-GPTProFixed"

    # QA-optimized training setup
    batch_size_per_gpu: int = 16
    effective_batch_target: int = 16
    num_workers: int = max(2, min(8, os.cpu_count() or 2))

    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.08
    max_grad_norm: float = 1.0

    amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999

    # Validation model selection for checkpointing:
    # - "student": always validate/select student model
    # - "ema": always validate/select EMA model (fallback to student if EMA is off)
    # - "best_of_both": compare student vs EMA on val each epoch and pick better
    val_selection_source: str = "best_of_both"

    text_dim: int = 256
    max_prompt_len: int = 12
    # Clean ablation switches mapped back to the original I1-I5 story:
    # I1: prompt normalization
    # I2: safe QaTa augmentation
    # I3: number of LV fusion stages (1 for I2-style, 4 for final)
    # I4: focal tversky loss
    # I5: image-text alignment loss
    use_i1_prompt_normalization: bool = True
    use_i2_safe_augmentation: bool = True
    num_lv_fusion_stages: int = 4
    use_i4_focal_tversky: bool = True
    use_i5_alignment_full: bool = True

    threshold_grid = np.arange(0.35, 0.76, 0.05)

    # Train by labeled percentage (NOT fixed sample count)
    label_percent: float = 50.0

    # Pseudo-label configuration (for unlabeled split)
    # enable_pseudo_label will be auto-set from label_percent below
    enable_pseudo_label: bool = True
    pseudo_confidence_threshold: float = 0.80
    pseudo_loss_weight: float = 0.30
    pseudo_warmup_epochs: int = 1
    # EPI momentum: P_t = beta * P_{t-1} + (1-beta) * P_t_tilde
    epi_beta: float = 0.99
    unlabeled_batch_size_per_gpu: int = 16

    # Resume training from last checkpoint
    resume_training: bool = False
    resume_checkpoint_path: str = ""  # empty -> <save_dir>/last_checkpoint.pth

    # Performance knobs
    train_metrics_every_n_steps: int = 16
    pseudo_every_n_steps: int = 3
    prefetch_factor: int = 4
    allow_tf32: bool = True
    channels_last: bool = True
    use_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"

    save_dir: str = "/kaggle/working/qata_lvit_nobert_50pct_gptpro_fixed_runs"
    dataset_hint_name: str = "QaTa-Covid19"
    epoch_metrics_csv_name: str = "epoch_metrics.csv"
    epoch_metrics_json_name: str = "epoch_metrics.json"

cfg = CFG()

def set_seed(seed=42):
    """Execute a documented step of the segmentation training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)
torch.backends.cudnn.benchmark = True

# runtime speed flags
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

if not (0.0 < float(cfg.label_percent) <= 100.0):
    raise ValueError(f"label_percent must be in (0, 100], got {cfg.label_percent}")

# Auto toggle pseudo-label based on label_percent:
# - label_percent < 100  -> enable_pseudo_label = True
# - label_percent == 100 -> enable_pseudo_label = False
cfg.enable_pseudo_label = bool(float(cfg.label_percent) < 100.0)

os.makedirs(cfg.save_dir, exist_ok=True)
print(asdict(cfg))

# %% code cell 2

# -----------------------------
# Dataset discovery / extraction
# -----------------------------
def discover_qata_root():
    """
    Locate the QaTa-Covid19 dataset in the most common Kaggle layouts.

    Priority order:
    1) already extracted inside /kaggle/input
    2) packaged as a zip file in /kaggle/input or /kaggle/working
    3) fallback to local paths such as /mnt/data
    """
    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("/mnt/data"),
        Path.cwd(),
    ]

    folder_candidates = []
    zip_candidates = []

    for root in search_roots:
        if not root.exists():
            continue

        for p in root.rglob(cfg.dataset_hint_name):
            if (p / "Train_Folder" / "img").exists():
                folder_candidates.append(p)

        for p in root.rglob(f"{cfg.dataset_hint_name}.zip"):
            zip_candidates.append(p)

    if folder_candidates:
        root = sorted(folder_candidates, key=lambda x: len(str(x)))[0]
        print(f"Found extracted dataset at: {root}")
        return root

    if zip_candidates:
        zip_path = sorted(zip_candidates, key=lambda x: len(str(x)))[0]
        extract_parent = Path(cfg.save_dir) / "extracted_data"
        extract_parent.mkdir(parents=True, exist_ok=True)
        target_root = extract_parent / cfg.dataset_hint_name

        if not (target_root / "Train_Folder" / "img").exists():
            print(f"Extracting {zip_path} -> {extract_parent}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_parent)

        if (target_root / "Train_Folder" / "img").exists():
            return target_root

        # Fall back if the extracted zip creates a nested folder with a slightly
        # different name.
        for p in extract_parent.rglob(cfg.dataset_hint_name):
            if (p / "Train_Folder" / "img").exists():
                return p

    raise FileNotFoundError("Could not find the QaTa-Covid19 folder or QaTa-Covid19.zip")

DATA_ROOT = discover_qata_root()
print("DATA_ROOT =", DATA_ROOT)

# %% code cell 3

# -----------------------------
# Prompt normalization (QaTa-aware)
# -----------------------------
NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}

REGION_ORDER = ["upper", "middle", "lower"]

def clean_text(s: str) -> str:
    """Execute a documented step of the segmentation training pipeline."""
    s = str(s).lower().strip()
    s = s.replace("andall", "and all")
    s = s.replace(",all", ", all")
    s = s.replace("mmiddle", "middle")
    s = s.replace("higher", "upper")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_side_regions(s: str, side: str):
    """Execute a documented step of the segmentation training pipeline."""
    matches = re.findall(r"((?:all|upper|middle|lower|\s)+)\s+" + side + r"\s+lung", s)
    tokens = set()
    for phrase in matches:
        phrase = clean_text(phrase)
        if "all" in phrase:
            return ["all"]
        for region in REGION_ORDER:
            if region in phrase:
                tokens.add(region)
    return [r for r in REGION_ORDER if r in tokens]

def normalize_prompt(desc: str):
    """
    Normalize a free-form report into a compact and stable prompt token list.

    Example:
      'Bilateral pulmonary infection, two infected areas, all left lung and middle lower right lung.'
    -> ['task_qatacovid19', 'lat_bilateral', 'count_2', 'left_all', 'right_middle', 'right_lower']
    """
    s = clean_text(desc)

    laterality = "bilateral" if "bilateral" in s else "unilateral"

    count = None
    for word, value in NUM_WORDS.items():
        if f"{word} infected area" in s or f"{word} infected areas" in s:
            count = value
            break

    if count is None:
        m = re.search(r"(\d+)\s+infected", s)
        count = int(m.group(1)) if m else 0

    left_regs = extract_side_regions(s, "left")
    right_regs = extract_side_regions(s, "right")

    tokens = ["task_qatacovid19", f"lat_{laterality}", f"count_{count}"]

    if left_regs:
        if left_regs == ["all"]:
            tokens.append("left_all")
        else:
            tokens.extend([f"left_{r}" for r in left_regs])
    else:
        tokens.append("left_none")

    if right_regs:
        if right_regs == ["all"]:
            tokens.append("right_all")
        else:
            tokens.extend([f"right_{r}" for r in right_regs])
    else:
        tokens.append("right_none")

    return tokens


def basic_prompt_tokens(desc: str):
    """Execute a documented step of the segmentation training pipeline."""
    s = clean_text(desc)
    tokens = s.split()
    if len(tokens) == 0:
        tokens = ["empty_prompt"]
    return tokens

def build_manifest(split_name: str):
    """Execute a documented step of the segmentation training pipeline."""
    split_dir = DATA_ROOT / split_name
    img_dir = split_dir / "img"
    mask_dir = split_dir / "labelcol"

    excel_files = list(split_dir.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"No xlsx file found in {split_dir}")

    df = pd.read_excel(excel_files[0])
    df = df.rename(columns={df.columns[0]: "image", df.columns[1]: "description"})
    df["image"] = df["image"].astype(str)
    df["description"] = df["description"].astype(str)

    # Quan trọng: val_text.xlsx trong một số bản phát hành có thể chứa toàn bộ danh sách mô tả,
    # nên ở đây luôn lọc theo ảnh thực sự tồn tại trong split hiện tại.
    img_set = {p.name for p in img_dir.glob("*.png")}
    mask_set = {p.name for p in mask_dir.glob("*.png")}
    valid = img_set & mask_set

    df = df[df["image"].isin(valid)].drop_duplicates("image").copy()
    df["image_path"] = df["image"].apply(lambda x: str(img_dir / x))
    df["mask_path"] = df["image"].apply(lambda x: str(mask_dir / x))
    prompt_fn = normalize_prompt if bool(getattr(cfg, "use_i1_prompt_normalization", True)) else basic_prompt_tokens
    df["prompt_tokens"] = df["description"].apply(prompt_fn)

    return df.reset_index(drop=True)

train_df = build_manifest("Train_Folder")
val_df   = build_manifest("Val_Folder")
test_df  = build_manifest("Test_Folder")

print("train:", train_df.shape)
print("val  :", val_df.shape)
print("test :", test_df.shape)
display(train_df.head(3)[["image", "description", "prompt_tokens"]])

# %% code cell 4

# -----------------------------
# Basic stats + vocabulary + mask area cache
# -----------------------------
def compute_mask_area(mask_path: str) -> float:
    """Execute a documented step of the segmentation training pipeline."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0
    return float((mask > 0).mean())

def attach_mask_area(df: pd.DataFrame, cache_name: str):
    """Execute a documented step of the segmentation training pipeline."""
    cache_path = Path(cfg.save_dir) / f"{cache_name}_mask_area.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        return df.merge(cached, on="image", how="left")

    out = df[["image", "mask_path"]].copy()
    out["mask_area"] = [compute_mask_area(p) for p in tqdm(out["mask_path"], total=len(out))]
    out[["image", "mask_area"]].to_csv(cache_path, index=False)
    return df.merge(out[["image", "mask_area"]], on="image", how="left")

train_df = attach_mask_area(train_df, "train")
val_df   = attach_mask_area(val_df, "val")

def build_vocab(prompt_lists):
    """Execute a documented step of the segmentation training pipeline."""
    counter = Counter()
    for toks in prompt_lists:
        counter.update(toks)

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common():
        vocab[token] = len(vocab)
    return vocab

vocab = build_vocab(train_df["prompt_tokens"].tolist())

def encode_prompt(tokens, vocab, max_len=12):
    """Execute a documented step of the segmentation training pipeline."""
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens][:max_len]
    attn = [1] * len(ids)

    while len(ids) < max_len:
        ids.append(vocab["<pad>"])
        attn.append(0)

    return np.array(ids, dtype=np.int64), np.array(attn, dtype=np.int64)

def estimate_mean_std(df, sample_n=512):
    """Execute a documented step of the segmentation training pipeline."""
    idxs = np.linspace(0, len(df) - 1, min(sample_n, len(df))).astype(int)
    means, stds = [], []
    for idx in tqdm(idxs, total=len(idxs), leave=False):
        img = cv2.imread(df.iloc[idx]["image_path"], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        means.append(img.mean())
        stds.append(img.std())
    return float(np.mean(means)), float(np.mean(stds) + 1e-6)

TRAIN_MEAN, TRAIN_STD = estimate_mean_std(train_df, sample_n=512)
POS_WEIGHT = min(3.0, max(1.0, float((1.0 - train_df["mask_area"].mean()) / max(train_df["mask_area"].mean(), 1e-6))))

print("vocab size:", len(vocab))
print("train mean/std:", TRAIN_MEAN, TRAIN_STD)
print("avg train mask area:", train_df["mask_area"].mean())
print("bce pos_weight:", POS_WEIGHT)

display(train_df["prompt_tokens"].apply(len).describe())
display(train_df["mask_area"].describe())

# %% code cell 5
# -----------------------------
# Safe augmentations for QaTa
# -----------------------------
class SafeQaTaAugment:
    """
    Khong dung horizontal/vertical flip hoac rot90 de khong pha semantics left/right.
    Chi dung affine nho + intensity augmentation.
    """
    def __init__(self, size=224):
        """Execute a documented step of the segmentation training pipeline."""
        self.size = size

    def _affine(self, image, mask):
        """Execute a documented step of the segmentation training pipeline."""
        h, w = image.shape

        angle = random.uniform(-7, 7)
        scale = random.uniform(0.96, 1.04)
        tx = random.uniform(-0.03, 0.03) * w
        ty = random.uniform(-0.02, 0.02) * h

        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        image = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        mask = cv2.warpAffine(
            mask, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT
        )
        return image, mask

    def _intensity(self, image):
        """Execute a documented step of the segmentation training pipeline."""
        if random.random() < 0.35:
            alpha = random.uniform(0.90, 1.10)
            beta = random.uniform(-0.05, 0.05)
            image = np.clip(image * alpha + beta, 0, 1)

        if random.random() < 0.25:
            gamma = random.uniform(0.90, 1.10)
            image = np.clip(image ** gamma, 0, 1)

        if random.random() < 0.20:
            sigma = random.uniform(0.0, 0.02)
            noise = np.random.normal(0, sigma, image.shape)
            image = np.clip(image + noise, 0, 1)

        if random.random() < 0.20:
            clahe = cv2.createCLAHE(clipLimit=random.uniform(1.0, 2.0), tileGridSize=(8, 8))
            image = clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255.0

        return image

    def __call__(self, image, mask):
        """Execute a documented step of the segmentation training pipeline."""
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        image, mask = self._affine(image, mask)
        image = self._intensity(image)

        return image, mask

class QaTaValTransform:
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, size=224):
        """Execute a documented step of the segmentation training pipeline."""
        self.size = size

    def __call__(self, image, mask):
        """Execute a documented step of the segmentation training pipeline."""
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return image, mask

class QaTaDataset(Dataset):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, df, vocab, transform=None, image_mean=0.5, image_std=0.25):
        """Execute a documented step of the segmentation training pipeline."""
        self.df = df.reset_index(drop=True).copy()
        self.vocab = vocab
        self.transform = transform
        self.image_mean = image_mean
        self.image_std = image_std

    def __len__(self):
        """Execute a documented step of the segmentation training pipeline."""
        return len(self.df)

    def __getitem__(self, idx):
        """Execute a documented step of the segmentation training pipeline."""
        row = self.df.iloc[idx]

        image = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = (image - self.image_mean) / self.image_std
        image = torch.tensor(image[None, ...], dtype=torch.float32)
        mask = torch.tensor(mask[None, ...], dtype=torch.float32)

        prompt_ids, prompt_mask = encode_prompt(row.prompt_tokens, self.vocab, cfg.max_prompt_len)

        return {
            "image": image,
            "mask": mask,
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "prompt_mask": torch.tensor(prompt_mask, dtype=torch.long),
            "image_name": row.image,
        }

def split_labeled_unlabeled_df(df: pd.DataFrame, label_percent: float, seed: int):
    """Execute a documented step of the segmentation training pipeline."""
    if len(df) == 0:
        raise ValueError("Empty train dataframe.")

    ratio = float(label_percent) / 100.0
    n_total = len(df)
    n_labeled = max(1, int(round(n_total * ratio)))
    n_labeled = min(n_total, n_labeled)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)

    labeled_idx = perm[:n_labeled]
    unlabeled_idx = perm[n_labeled:]

    labeled_df = df.iloc[labeled_idx].reset_index(drop=True)
    unlabeled_df = df.iloc[unlabeled_idx].reset_index(drop=True)
    return labeled_df, unlabeled_df

def make_area_balanced_sampler(df: pd.DataFrame):
    """Execute a documented step of the segmentation training pipeline."""
    if len(df) == 0:
        return None
    uniq = int(df["mask_area"].nunique())
    if uniq < 2:
        weights = np.ones(len(df), dtype=np.float32)
    else:
        bins = pd.qcut(df["mask_area"], q=min(5, uniq), duplicates="drop")
        counts = bins.value_counts().to_dict()
        weights = bins.map(lambda x: 1.0 / counts[x]).astype(np.float32).values
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )

train_labeled_df, train_unlabeled_df = split_labeled_unlabeled_df(
    train_df,
    label_percent=cfg.label_percent,
    seed=cfg.seed,
)

train_transform = (
    SafeQaTaAugment(cfg.image_size)
    if bool(getattr(cfg, "use_i2_safe_augmentation", True))
    else QaTaValTransform(cfg.image_size)
)

train_ds = QaTaDataset(
    train_labeled_df, vocab,
    transform=train_transform,
    image_mean=TRAIN_MEAN,
    image_std=TRAIN_STD,
)

# Unlabeled split: keep image + text, do not use mask for supervision
# Use deterministic/weak transform to keep content stable for pseudo-labeling.
unlabeled_ds = None
if len(train_unlabeled_df) > 0:
    unlabeled_ds = QaTaDataset(
        train_unlabeled_df, vocab,
        transform=QaTaValTransform(cfg.image_size),
        image_mean=TRAIN_MEAN,
        image_std=TRAIN_STD,
    )

val_ds = QaTaDataset(
    val_df, vocab,
    transform=QaTaValTransform(cfg.image_size),
    image_mean=TRAIN_MEAN,
    image_std=TRAIN_STD,
)

test_ds = QaTaDataset(
    test_df, vocab,
    transform=QaTaValTransform(cfg.image_size),
    image_mean=TRAIN_MEAN,
    image_std=TRAIN_STD,
)

# Keep this for optional experiments, but default training now uses plain shuffle
train_sampler = None

print(f"label_percent={cfg.label_percent:.2f}%")
print(f"train total={len(train_df)} | labeled={len(train_labeled_df)} | unlabeled={len(train_unlabeled_df)}")

# quick sanity visualization
sample = train_ds[0]
print(sample["image"].shape, sample["mask"].shape, sample["prompt_ids"].shape)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample["image"][0].numpy(), cmap="gray")
plt.title("image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(sample["mask"][0].numpy(), cmap="gray")
plt.title("mask")
plt.axis("off")
plt.show()

# %% code cell 6

# -----------------------------
# LViT-style model
# -----------------------------
class ConvBNAct(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """Execute a documented step of the segmentation training pipeline."""
        return self.block(x)

class DoubleConv(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, in_ch, out_ch):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1, 1),
            ConvBNAct(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x):
        """Execute a documented step of the segmentation training pipeline."""
        return self.net(x)

class Down(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, in_ch, out_ch):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        """Execute a documented step of the segmentation training pipeline."""
        return self.net(x)

class Up(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, in_ch, skip_ch, out_ch):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        """Execute a documented step of the segmentation training pipeline."""
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MLP(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Execute a documented step of the segmentation training pipeline."""
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class TransformerBlock(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        """Execute a documented step of the segmentation training pipeline."""
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class PromptEncoder(nn.Module):
    """
    Text encoder nhẹ, học trực tiếp trên prompt đã chuẩn hóa cho QaTa.
    """
    def __init__(self, vocab_size, emb_dim=96, hidden_dim=192, out_dim=256, dropout=0.1):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, ids, mask):
        """Execute a documented step of the segmentation training pipeline."""
        emb = self.embedding(ids)
        lengths = mask.sum(1).clamp(min=1).cpu()

        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=ids.size(1))

        out = self.norm(self.proj(self.dropout(out)))
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        return out, pooled

class PatchTokenizer(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, in_ch, embed_dim, patch_size, img_size):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Execute a documented step of the segmentation training pipeline."""
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x + self.pos_embed)
        return x

class TokenToSpatial(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, dim, out_ch, scale_factor):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = nn.Sequential(
            nn.Conv2d(dim, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, tokens):
        """Execute a documented step of the segmentation training pipeline."""
        b, n, c = tokens.shape
        h = w = int(math.sqrt(n))
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return self.proj(x)

class CrossLanguageFusion(nn.Module):
    """
    Cross-attention text -> image tokens ở mỗi scale, thay vì chỉ fuse ở stage đầu.
    """
    def __init__(self, dim, text_dim, num_heads=4, drop=0.0):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.t_norm = nn.LayerNorm(text_dim)
        self.text_proj = nn.Linear(text_dim, dim)

        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=drop, batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x_tokens, text_tokens, text_mask=None):
        """Execute a documented step of the segmentation training pipeline."""
        t = self.text_proj(self.t_norm(text_tokens))
        q = self.q_norm(x_tokens)

        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask.bool()

        attn_out = self.cross_attn(
            q, t, t,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

        t_mean = t.mean(dim=1, keepdim=True).expand_as(attn_out)
        gate = self.gate(torch.cat([q, t_mean], dim=-1))

        return x_tokens + gate * attn_out

class LVFusionStage(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, in_ch, token_dim, patch_size, img_size, text_dim, depth=1, num_heads=4, drop=0.0):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.tokenizer = PatchTokenizer(in_ch, token_dim, patch_size, img_size)
        self.cross = CrossLanguageFusion(token_dim, text_dim, num_heads=num_heads, drop=drop)
        self.blocks = nn.Sequential(*[
            TransformerBlock(token_dim, num_heads=num_heads, mlp_ratio=4.0, drop=drop)
            for _ in range(depth)
        ])
        self.reconstruct = TokenToSpatial(token_dim, in_ch, scale_factor=patch_size)

    def forward(self, feat, text_tokens, text_mask):
        """Execute a documented step of the segmentation training pipeline."""
        tokens = self.tokenizer(feat)
        tokens = self.cross(tokens, text_tokens, text_mask)
        tokens = self.blocks(tokens)
        recon = self.reconstruct(tokens)
        return feat + recon, tokens

class QaTaLViT(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, vocab_size, in_ch=1, text_dim=256, num_classes=1, num_lv_fusion_stages=4):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.num_lv_fusion_stages = int(max(1, min(4, num_lv_fusion_stages)))

        self.text_encoder = PromptEncoder(vocab_size=vocab_size, out_dim=text_dim)

        # image encoder
        self.stem  = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bottom = Down(512, 512)

        # multi-scale LV fusion
        self.lv1 = LVFusionStage(64,  64,  patch_size=16, img_size=224, text_dim=text_dim, depth=1, num_heads=4, drop=0.05)
        self.lv2 = LVFusionStage(128, 128, patch_size=8,  img_size=112, text_dim=text_dim, depth=1, num_heads=4, drop=0.05)
        self.lv3 = LVFusionStage(256, 256, patch_size=4,  img_size=56,  text_dim=text_dim, depth=2, num_heads=8, drop=0.05)
        self.lv4 = LVFusionStage(512, 512, patch_size=2,  img_size=28,  text_dim=text_dim, depth=2, num_heads=8, drop=0.05)

        # decoder
        self.up4 = Up(512, 512, 256)
        self.up3 = Up(256, 256, 128)
        self.up2 = Up(128, 128, 64)
        self.up1 = Up(64,  64,  64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

        # auxiliary LV alignment
        self.img_proj = nn.Linear(512, 256)
        self.text_proj = nn.Linear(text_dim, 256)

    def forward(self, image, prompt_ids, prompt_mask):
        """Execute a documented step of the segmentation training pipeline."""
        text_tokens, text_global = self.text_encoder(prompt_ids, prompt_mask)

        x1 = self.stem(image)
        if self.num_lv_fusion_stages >= 1:
            e1, _ = self.lv1(x1, text_tokens, prompt_mask)
        else:
            e1 = x1

        x2 = self.down1(e1)
        if self.num_lv_fusion_stages >= 2:
            e2, _ = self.lv2(x2, text_tokens, prompt_mask)
        else:
            e2 = x2

        x3 = self.down2(e2)
        if self.num_lv_fusion_stages >= 3:
            e3, _ = self.lv3(x3, text_tokens, prompt_mask)
        else:
            e3 = x3

        x4 = self.down3(e3)
        if self.num_lv_fusion_stages >= 4:
            e4, _ = self.lv4(x4, text_tokens, prompt_mask)
        else:
            e4 = x4

        x5 = self.bottom(e4)

        x = self.up4(x5, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        logits = self.head(x)

        img_global = self.img_proj(F.adaptive_avg_pool2d(x5, 1).flatten(1))
        text_global = self.text_proj(text_global)

        return {
            "logits": logits,
            "img_embed": img_global,
            "text_embed": text_global,
        }

# %% code cell 7

# -----------------------------
# Losses, metrics, EMA, loaders
# -----------------------------
def dice_loss_with_logits(logits, targets, eps=1e-6):
    """Execute a documented step of the segmentation training pipeline."""
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()

def focal_tversky_loss(logits, targets, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
    """Execute a documented step of the segmentation training pipeline."""
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)

    tp = (probs * targets).sum(dims)
    fp = (probs * (1 - targets)).sum(dims)
    fn = ((1 - probs) * targets).sum(dims)

    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return ((1 - tversky) ** gamma).mean()

def clip_alignment_loss(img_embed, text_embed, temperature=0.07):
    """Execute a documented step of the segmentation training pipeline."""
    if img_embed.size(0) < 2:
        return img_embed.new_tensor(0.0)

    img = F.normalize(img_embed, dim=-1)
    txt = F.normalize(text_embed, dim=-1)
    logits = img @ txt.T / temperature
    labels = torch.arange(img.size(0), device=img.device)

    return 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    )

class QaTaLoss(nn.Module):
    """Implementation class used by the segmentation training pipeline."""
    def __init__(
        self,
        pos_weight=1.0,
        bce_w=0.45,
        dice_w=0.35,
        tv_w=0.15,
        align_w=0.05,
        use_i4_focal_tversky=True,
        use_i5_alignment_full=True,
    ):
        """Execute a documented step of the segmentation training pipeline."""
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.tv_w = tv_w
        self.align_w = align_w
        self.use_i4_focal_tversky = bool(use_i4_focal_tversky)
        self.use_i5_alignment_full = bool(use_i5_alignment_full)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))

    def forward(self, outputs, targets):
        """Execute a documented step of the segmentation training pipeline."""
        logits = outputs["logits"]

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
        )
        dice = dice_loss_with_logits(logits, targets)
        tv = focal_tversky_loss(logits, targets) if self.use_i4_focal_tversky else logits.new_tensor(0.0)
        align = (
            clip_alignment_loss(outputs["img_embed"], outputs["text_embed"])
            if self.use_i5_alignment_full
            else logits.new_tensor(0.0)
        )

        loss = (
            self.bce_w * bce +
            self.dice_w * dice +
            self.tv_w * tv +
            self.align_w * align
        )

        stats = {
            "loss": float(loss.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "dice_loss": float(dice.detach().cpu()),
            "tversky": float(tv.detach().cpu()),
            "align": float(align.detach().cpu()),
        }
        return loss, stats

def batch_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    """Execute a documented step of the segmentation training pipeline."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets > 0.5).float()

    dims = (1, 2, 3)
    inter = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)

    dice = ((2 * inter + eps) / (union + eps)).mean().item()
    iou = ((inter + eps) / (preds.sum(dims) + targets.sum(dims) - inter + eps)).mean().item()
    return dice, iou

class ModelEMA:
    """Implementation class used by the segmentation training pipeline."""
    def __init__(self, model, decay=0.999):
        """Execute a documented step of the segmentation training pipeline."""
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """Execute a documented step of the segmentation training pipeline."""
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k not in msd:
                continue
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])

def unwrap_model(model):
    """Execute a documented step of the segmentation training pipeline."""
    return model.module if isinstance(model, nn.DataParallel) else model

def make_loader(dataset, batch_size, sampler=None, shuffle=False, drop_last=False):
    """Execute a documented step of the segmentation training pipeline."""
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=drop_last,
    )
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(max(2, cfg.prefetch_factor))
    return DataLoader(**loader_kwargs)

def make_scheduler(optimizer, total_steps, warmup_ratio=0.08):
    """Execute a documented step of the segmentation training pipeline."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        """Execute a documented step of the segmentation training pipeline."""
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def autocast_context(device, enabled=True):
    """Execute a documented step of the segmentation training pipeline."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()

def get_runtime():
    """Execute a documented step of the segmentation training pipeline."""
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        usable_gpus = min(n_gpu, 2)
        device = torch.device("cuda")
        print(f"CUDA available | total visible GPUs = {n_gpu} | using = {usable_gpus}")
    else:
        usable_gpus = 0
        device = torch.device("cpu")
        print("CUDA not available, fallback to CPU")

    return device, usable_gpus

def build_model(vocab_size):
    """Execute a documented step of the segmentation training pipeline."""
    device, usable_gpus = get_runtime()
    model = QaTaLViT(
        vocab_size=vocab_size,
        num_lv_fusion_stages=int(getattr(cfg, "num_lv_fusion_stages", 4)),
    )

    if device.type == "cuda" and usable_gpus >= 2:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=list(range(usable_gpus)))
        print("Running with DataParallel over 2 GPUs")
    elif device.type == "cuda":
        model = model.cuda()
        print("Running on single GPU")
    else:
        print("Running on CPU")

    return device, model, usable_gpus

# %% code cell 8
# -----------------------------
# Build model / optimizer / loaders
# -----------------------------
device, model, usable_gpus = build_model(len(vocab))

# memory format / compiler optimizations
if cfg.channels_last and device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if bool(cfg.use_torch_compile) and hasattr(torch, "compile") and not isinstance(model, nn.DataParallel):
    try:
        model = torch.compile(model, mode=str(cfg.torch_compile_mode), fullgraph=False)
        print(f"torch.compile enabled (mode={cfg.torch_compile_mode})")
    except Exception as e:
        print(f"torch.compile skipped due to: {e}")

if len(train_ds) <= 0:
    raise ValueError("No labeled samples available for training.")

effective_gpus = max(1, usable_gpus if usable_gpus > 0 else 1)
batch_size = cfg.batch_size_per_gpu * effective_gpus
unlabeled_batch_size = cfg.unlabeled_batch_size_per_gpu * effective_gpus
grad_accum_steps = max(1, math.ceil(cfg.effective_batch_target / batch_size))

# Labeled loader: use full labeled subset each epoch (shuffle only, no replacement sampler)
train_loader = make_loader(
    train_ds,
    batch_size=batch_size,
    sampler=None,
    shuffle=True,
    drop_last=False,
)

# Unlabeled loader: keep all remaining image+text for pseudo-label generation
if cfg.enable_pseudo_label and (unlabeled_ds is not None) and (len(unlabeled_ds) > 0):
    unlabeled_loader = make_loader(
        unlabeled_ds,
        batch_size=unlabeled_batch_size,
        sampler=None,
        shuffle=True,
        drop_last=False,
    )
else:
    unlabeled_loader = None

val_loader = make_loader(
    val_ds,
    batch_size=batch_size,
    sampler=None,
    shuffle=False,
    drop_last=False,
)

test_loader = make_loader(
    test_ds,
    batch_size=batch_size,
    sampler=None,
    shuffle=False,
    drop_last=False,
)

criterion = QaTaLoss(
    pos_weight=POS_WEIGHT,
    use_i4_focal_tversky=bool(getattr(cfg, "use_i4_focal_tversky", True)),
    use_i5_alignment_full=bool(getattr(cfg, "use_i5_alignment_full", True)),
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)

labeled_steps = len(train_loader)
unlabeled_steps = len(unlabeled_loader) if unlabeled_loader is not None else 0
steps_per_epoch_batches = max(labeled_steps, unlabeled_steps, 1)
steps_per_epoch_optim = math.ceil(steps_per_epoch_batches / max(1, grad_accum_steps))
total_steps = cfg.epochs * steps_per_epoch_optim
scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_ratio=cfg.warmup_ratio)

use_amp = bool(cfg.amp and device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
ema = ModelEMA(unwrap_model(model), decay=cfg.ema_decay) if cfg.use_ema else None

resume_start_epoch = 1
resume_best_dice = -1.0
resume_loaded_from = None

if bool(getattr(cfg, "resume_training", False)):
    candidate = str(getattr(cfg, "resume_checkpoint_path", "")).strip()
    resume_ckpt_path = Path(candidate) if candidate else (Path(cfg.save_dir) / "last_checkpoint.pth")
    if resume_ckpt_path.exists():
        ckpt = torch.load(str(resume_ckpt_path), map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        if ema is not None and ("ema" in ckpt):
            ema.ema.load_state_dict(ckpt["ema"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ("scaler" in ckpt) and (scaler is not None):
            scaler.load_state_dict(ckpt["scaler"])

        resume_start_epoch = int(ckpt.get("epoch", 0)) + 1
        resume_best_dice = float(ckpt.get("best_dice", -1.0))
        resume_loaded_from = str(resume_ckpt_path)
        print(f"Resumed training from: {resume_loaded_from}")
        print(f"resume_start_epoch={resume_start_epoch}, resume_best_dice={resume_best_dice:.4f}")
    else:
        print(f"resume_training=True but checkpoint not found at: {resume_ckpt_path}")

print("batch_size (labeled):", batch_size)
print("batch_size (unlabeled):", unlabeled_batch_size)
print("grad_accum_steps:", grad_accum_steps)
print("labeled steps/epoch:", labeled_steps)
print("unlabeled steps/epoch:", unlabeled_steps)
print("batches per epoch used in train loop:", steps_per_epoch_batches)
print("optimizer steps/epoch:", steps_per_epoch_optim)
print("total scheduler steps:", total_steps)
print("pseudo_label enabled:", bool(unlabeled_loader is not None))
print("resume_training:", bool(getattr(cfg, "resume_training", False)))
if resume_loaded_from is not None:
    print("resume_loaded_from:", resume_loaded_from)

# Sanity forward pass
batch = next(iter(train_loader))
images = batch["image"].to(device, non_blocking=True)
masks = batch["mask"].to(device, non_blocking=True)
prompt_ids = batch["prompt_ids"].to(device, non_blocking=True)
prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)

with torch.no_grad():
    outputs = model(images, prompt_ids, prompt_mask)

print("logits:", outputs["logits"].shape)
print("img_embed:", outputs["img_embed"].shape)
print("text_embed:", outputs["text_embed"].shape)

# %% code cell 9
# -----------------------------
# Train / eval loops
# -----------------------------
def _next_batch(batch_iter, loader):
    """Execute a documented step of the segmentation training pipeline."""
    try:
        batch = next(batch_iter)
    except StopIteration:
        batch_iter = iter(loader)
        batch = next(batch_iter)
    return batch, batch_iter


def _epi_blend_probs(epi_state, image_names, current_probs, beta):
    """Execute a documented step of the segmentation training pipeline."""
    beta = float(beta)
    blended = current_probs.detach().clone()

    for b, name in enumerate(image_names):
        key = str(name)
        prev = epi_state.get(key, None)
        cur = current_probs[b:b+1]

        if prev is not None:
            prev = prev.to(device=current_probs.device, dtype=current_probs.dtype)
            if prev.ndim == 3:
                prev = prev.unsqueeze(0)
            if prev.shape[-2:] != cur.shape[-2:]:
                prev = F.interpolate(prev, size=cur.shape[-2:], mode="bilinear", align_corners=False)
            if prev.shape != cur.shape:
                prev = prev.reshape_as(cur)
            blended[b:b+1] = beta * prev + (1.0 - beta) * cur

        epi_state[key] = blended[b:b+1].detach().to("cpu", dtype=torch.float16).squeeze(0)

    return blended


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    device,
    grad_accum_steps=1,
    ema=None,
    unlabeled_loader=None,
    epoch=1,
    epi_state=None,
):
    """Execute a documented step of the segmentation training pipeline."""
    model.train()

    if epi_state is None:
        epi_state = {}

    unlabeled_enabled = (
        unlabeled_loader is not None
        and cfg.enable_pseudo_label
        and (epoch >= int(cfg.pseudo_warmup_epochs))
    )

    labeled_steps = len(loader)
    unlabeled_steps = len(unlabeled_loader) if unlabeled_enabled else 0
    total_steps = max(labeled_steps, unlabeled_steps, 1)
    metrics_every = int(max(1, getattr(cfg, "train_metrics_every_n_steps", 4)))
    pseudo_every = int(max(1, getattr(cfg, "pseudo_every_n_steps", 1)))

    labeled_iter = iter(loader)
    unlabeled_iter = iter(unlabeled_loader) if unlabeled_enabled else None

    running = {
        "loss": 0.0,
        "sup_loss": 0.0,
        "lv_loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "tversky": 0.0,
        "align": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "pseudo_valid_ratio": 0.0,
    }

    optimizer.zero_grad(set_to_none=True)
    metric_steps = 0
    pbar = tqdm(range(1, total_steps + 1), leave=False)

    for step in pbar:
        batch, labeled_iter = _next_batch(labeled_iter, loader)

        images = batch["image"].to(device, non_blocking=True)
        if cfg.channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        masks = batch["mask"].to(device, non_blocking=True)
        prompt_ids = batch["prompt_ids"].to(device, non_blocking=True)
        prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)

        with autocast_context(device, enabled=use_amp):
            outputs = model(images, prompt_ids, prompt_mask)
            sup_loss, stats = criterion(outputs, masks)

            dice_batch = 0.0
            iou_batch = 0.0
            do_metrics = (step % metrics_every == 0) or (step == total_steps)
            if do_metrics:
                dice_batch, iou_batch = batch_metrics_from_logits(outputs["logits"], masks, threshold=0.5)

            lv_loss = outputs["logits"].mean() * 0.0
            pseudo_valid_ratio = 0.0

            pseudo_this_step = unlabeled_enabled and (step % pseudo_every == 0)
            if pseudo_this_step:
                ubatch, unlabeled_iter = _next_batch(unlabeled_iter, unlabeled_loader)

                u_images = ubatch["image"].to(device, non_blocking=True)
                if cfg.channels_last and u_images.ndim == 4:
                    u_images = u_images.contiguous(memory_format=torch.channels_last)
                u_prompt_ids = ubatch["prompt_ids"].to(device, non_blocking=True)
                u_prompt_mask = ubatch["prompt_mask"].to(device, non_blocking=True)

                teacher_model = ema.ema if ema is not None else unwrap_model(model)
                teacher_prev_mode = teacher_model.training
                teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = teacher_model(u_images, u_prompt_ids, u_prompt_mask)
                    teacher_probs = torch.sigmoid(teacher_outputs["logits"])
                if teacher_prev_mode:
                    teacher_model.train()

                u_names = ubatch["image_name"]
                epi_probs = _epi_blend_probs(
                    epi_state=epi_state,
                    image_names=u_names,
                    current_probs=teacher_probs,
                    beta=float(getattr(cfg, "epi_beta", 0.99)),
                )

                pseudo_targets = (epi_probs >= 0.5).float()
                confidence = torch.maximum(epi_probs, 1.0 - epi_probs)
                valid_mask = (confidence >= float(cfg.pseudo_confidence_threshold)).float()
                pseudo_valid_ratio = float(valid_mask.mean().detach().cpu())

                u_outputs = model(u_images, u_prompt_ids, u_prompt_mask)
                u_logits = u_outputs["logits"]
                pseudo_bce_map = F.binary_cross_entropy_with_logits(u_logits, pseudo_targets, reduction="none")
                valid_pixels = valid_mask.sum()

                if float(valid_pixels.detach().cpu()) > 0.0:
                    pseudo_core = (pseudo_bce_map * valid_mask).sum() / valid_pixels
                else:
                    pseudo_core = u_logits.mean() * 0.0

                lv_loss = float(cfg.pseudo_loss_weight) * pseudo_core

            total_loss = sup_loss + lv_loss
            loss = total_loss / grad_accum_steps

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updated = scaler.get_scale() >= prev_scale
            if optimizer_updated:
                scheduler.step()
                if ema is not None:
                    ema.update(unwrap_model(model))

        running["loss"] += float(total_loss.detach().cpu())
        running["sup_loss"] += float(sup_loss.detach().cpu())
        running["lv_loss"] += float(lv_loss.detach().cpu())
        running["bce"] += float(stats["bce"])
        running["dice_loss"] += float(stats["dice_loss"])
        running["tversky"] += float(stats["tversky"])
        running["align"] += float(stats["align"])
        if do_metrics:
            running["dice"] += float(dice_batch)
            running["iou"] += float(iou_batch)
            metric_steps += 1
        running["pseudo_valid_ratio"] += float(pseudo_valid_ratio)

        pbar.set_postfix(
            loss=f"{running['loss'] / step:.4f}",
            sup=f"{running['sup_loss'] / step:.4f}",
            lv=f"{running['lv_loss'] / step:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    # flush last partial accumulation
    if total_steps % grad_accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        prev_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_updated = scaler.get_scale() >= prev_scale
        if optimizer_updated:
            scheduler.step()
            if ema is not None:
                ema.update(unwrap_model(model))

    for k in running:
        if k in ("dice", "iou"):
            running[k] /= max(1, metric_steps)
        else:
            running[k] /= total_steps

    return running

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, criterion=None):
    """Execute a documented step of the segmentation training pipeline."""
    model.eval()
    dice_sum = 0.0
    iou_sum = 0.0
    loss_sum = 0.0
    n_samples = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        if cfg.channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        masks = batch["mask"].to(device, non_blocking=True)
        prompt_ids = batch["prompt_ids"].to(device, non_blocking=True)
        prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)

        outputs = model(images, prompt_ids, prompt_mask)
        logits = outputs["logits"]

        bs = int(images.shape[0])
        dice, iou = batch_metrics_from_logits(logits, masks, threshold=threshold)
        dice_sum += float(dice) * bs
        iou_sum += float(iou) * bs
        n_samples += bs

        if criterion is not None:
            val_loss, _ = criterion(outputs, masks)
            loss_sum += float(val_loss.detach().cpu()) * bs

    out = {
        "dice": float(dice_sum / max(1, n_samples)),
        "iou": float(iou_sum / max(1, n_samples)),
    }

    if criterion is not None:
        out["loss"] = float(loss_sum / max(1, n_samples))
        out["sup_loss"] = out["loss"]
        out["lv_loss"] = 0.0

    return out

@torch.no_grad()
def tune_threshold(model, loader, device, thresholds):
    """Execute a documented step of the segmentation training pipeline."""
    rows = []
    best_thr = 0.5
    best_dice = -1.0

    for thr in thresholds:
        metrics = evaluate(model, loader, device, threshold=float(thr))
        rows.append({
            "threshold": float(thr),
            "dice": metrics["dice"],
            "iou": metrics["iou"],
        })

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_thr = float(thr)

    return best_thr, pd.DataFrame(rows)

def save_checkpoint(
    path,
    epoch,
    model_state,
    best_dice,
    vocab,
    cfg,
    optimizer_state=None,
    scheduler_state=None,
    scaler_state=None,
    ema_state=None,
    extra_meta=None,
):
    """Execute a documented step of the segmentation training pipeline."""
    ckpt = {
        "epoch": int(epoch),
        "model": model_state,
        "best_dice": float(best_dice),
        "vocab": vocab,
        "cfg": asdict(cfg),
    }
    if optimizer_state is not None:
        ckpt["optimizer"] = optimizer_state
    if scheduler_state is not None:
        ckpt["scheduler"] = scheduler_state
    if scaler_state is not None:
        ckpt["scaler"] = scaler_state
    if ema_state is not None:
        ckpt["ema"] = ema_state
    if isinstance(extra_meta, dict) and len(extra_meta) > 0:
        ckpt["meta"] = extra_meta
    torch.save(ckpt, path)

# %% code cell 10
# -----------------------------
# Main training
# -----------------------------
history = []
best_dice = float(globals().get("resume_best_dice", -1.0))
start_epoch = int(globals().get("resume_start_epoch", 1))
best_path = str(Path(cfg.save_dir) / "qata_lvit_best.pth")
last_path = str(Path(cfg.save_dir) / "last_checkpoint.pth")

epoch_metrics_csv = Path(cfg.save_dir) / cfg.epoch_metrics_csv_name
epoch_metrics_json = Path(cfg.save_dir) / cfg.epoch_metrics_json_name

best_model_source = "unknown"
best_epoch = int(max(0, start_epoch - 1))

if start_epoch > 1 and epoch_metrics_csv.exists():
    try:
        history = pd.read_csv(epoch_metrics_csv).to_dict(orient="records")
        if len(history) > 0:
            _hist_df = pd.DataFrame(history)
            if "val_dice" in _hist_df.columns and _hist_df["val_dice"].notna().any():
                _hist_df["val_dice"] = pd.to_numeric(_hist_df["val_dice"], errors="coerce")
                if _hist_df["val_dice"].notna().any():
                    _best_idx = _hist_df["val_dice"].idxmax()
                    best_epoch = int(_hist_df.loc[_best_idx, "epoch"])
                    if "val_model_source" in _hist_df.columns:
                        best_model_source = str(_hist_df.loc[_best_idx, "val_model_source"])
    except Exception:
        history = []

# EPI state bank for unlabeled pseudo-label smoothing across epochs
epi_state = {}

if start_epoch > cfg.epochs:
    print(f"Checkpoint epoch already reached target epochs (start_epoch={start_epoch}, cfg.epochs={cfg.epochs}).")


def _finite_or_neg_inf(x):
    """Execute a documented step of the segmentation training pipeline."""
    try:
        v = float(x)
    except Exception:
        return -1e9
    return v if np.isfinite(v) else -1e9


def _select_val_metrics(student_metrics, ema_metrics):
    """Execute a documented step of the segmentation training pipeline."""
    source_cfg = str(getattr(cfg, "val_selection_source", "best_of_both")).lower().strip()
    if source_cfg not in ("student", "ema", "best_of_both"):
        source_cfg = "best_of_both"

    if source_cfg == "student":
        return "student", student_metrics

    if source_cfg == "ema":
        if ema_metrics is not None:
            return "ema", ema_metrics
        return "student", student_metrics

    # best_of_both
    if ema_metrics is None:
        return "student", student_metrics

    student_dice = _finite_or_neg_inf(student_metrics.get("dice", np.nan))
    ema_dice = _finite_or_neg_inf(ema_metrics.get("dice", np.nan))
    if ema_dice > student_dice:
        return "ema", ema_metrics
    return "student", student_metrics


for epoch in range(start_epoch, cfg.epochs + 1):
    train_stats = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        criterion=criterion,
        device=device,
        grad_accum_steps=grad_accum_steps,
        ema=ema,
        unlabeled_loader=unlabeled_loader,
        epoch=epoch,
        epi_state=epi_state,
    )

    student_model = unwrap_model(model)
    val_metrics_student = evaluate(student_model, val_loader, device=device, threshold=0.5, criterion=criterion)

    val_metrics_ema = None
    if ema is not None:
        val_metrics_ema = evaluate(ema.ema, val_loader, device=device, threshold=0.5, criterion=criterion)

    selected_source, val_metrics = _select_val_metrics(val_metrics_student, val_metrics_ema)

    row = {
        "epoch": int(epoch),
        "label_percent": float(cfg.label_percent),
        "labeled_samples": int(len(train_labeled_df)),
        "unlabeled_samples": int(len(train_unlabeled_df)),

        # requested train metrics
        "loss": float(train_stats["loss"]),
        "sup_loss": float(train_stats["sup_loss"]),
        "lv_loss": float(train_stats["lv_loss"]),
        "dice": float(train_stats["dice"]),
        "iou": float(train_stats["iou"]),

        # selected validation metrics (used for best-checkpoint logic)
        "val_model_source": str(selected_source),
        "val_loss": float(val_metrics.get("loss", np.nan)),
        "val_sup_loss": float(val_metrics.get("sup_loss", np.nan)),
        "val_lv_loss": float(val_metrics.get("lv_loss", np.nan)),
        "val_dice": float(val_metrics.get("dice", np.nan)),
        "val_iou": float(val_metrics.get("iou", np.nan)),

        # debug visibility: both student/ema validation tracks
        "val_dice_student": float(val_metrics_student.get("dice", np.nan)),
        "val_iou_student": float(val_metrics_student.get("iou", np.nan)),
        "val_dice_ema": float(val_metrics_ema.get("dice", np.nan)) if val_metrics_ema is not None else np.nan,
        "val_iou_ema": float(val_metrics_ema.get("iou", np.nan)) if val_metrics_ema is not None else np.nan,

        # pseudo-label diagnostics
        "train_pseudo_valid_ratio": float(train_stats["pseudo_valid_ratio"]),
        "epi_bank_size": int(len(epi_state)),
    }
    history.append(row)

    # save epoch metrics each epoch
    history_df = pd.DataFrame(history)
    history_df.to_csv(epoch_metrics_csv, index=False)
    epoch_metrics_json.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    # Snapshot states
    student_model_state = copy.deepcopy(student_model.state_dict())
    ema_model_state = copy.deepcopy(ema.ema.state_dict()) if (ema is not None) else None
    selected_model_state = ema_model_state if (selected_source == "ema" and ema_model_state is not None) else student_model_state

    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
    scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None

    # Save LAST checkpoint every epoch for resume (always student + full training states)
    save_checkpoint(
        last_path,
        epoch=epoch,
        model_state=student_model_state,
        best_dice=best_dice,
        vocab=vocab,
        cfg=cfg,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        scaler_state=scaler_state,
        ema_state=ema_model_state,
        extra_meta={
            "selected_val_source": str(selected_source),
            "val_dice_selected": float(row["val_dice"]),
            "val_dice_student": float(row["val_dice_student"]),
            "val_dice_ema": float(row["val_dice_ema"]) if np.isfinite(row["val_dice_ema"]) else None,
        },
    )

    candidate_dice_raw = float(val_metrics.get("dice", np.nan))
    candidate_dice = _finite_or_neg_inf(candidate_dice_raw)
    best_exists = Path(best_path).exists()
    should_update_best = (not best_exists) or (candidate_dice > (best_dice + 1e-12))

    if should_update_best:
        if np.isfinite(candidate_dice_raw):
            best_dice = candidate_dice_raw
        best_model_source = str(selected_source)
        best_epoch = int(epoch)
        save_checkpoint(
            best_path,
            epoch=epoch,
            model_state=selected_model_state,
            best_dice=best_dice,
            vocab=vocab,
            cfg=cfg,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            scaler_state=scaler_state,
            ema_state=ema_model_state,
            extra_meta={
                "best_model_source": str(selected_source),
                "val_selection_source_cfg": str(getattr(cfg, "val_selection_source", "best_of_both")),
                "val_dice_selected": float(row["val_dice"]),
                "val_dice_student": float(row["val_dice_student"]),
                "val_dice_ema": float(row["val_dice_ema"]) if np.isfinite(row["val_dice_ema"]) else None,
            },
        )

    print(
        f"Epoch {epoch:03d} | "
        f"loss={row['loss']:.4f} | "
        f"sup_loss={row['sup_loss']:.4f} | "
        f"lv_loss={row['lv_loss']:.4f} | "
        f"dice={row['dice']:.4f} | "
        f"iou={row['iou']:.4f} | "
        f"val_src={row['val_model_source']} | "
        f"val_dice={row['val_dice']:.4f} | "
        f"val_iou={row['val_iou']:.4f} | "
        f"val_dice_student={row['val_dice_student']:.4f} | "
        f"val_dice_ema={row['val_dice_ema']:.4f} | "
        f"best_dice={best_dice:.4f}"
    )

history_df = pd.DataFrame(history)
display(history_df.tail())

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_df["epoch"], history_df["loss"], label="train_loss")
plt.plot(history_df["epoch"], history_df["sup_loss"], label="train_sup_loss")
plt.plot(history_df["epoch"], history_df["lv_loss"], label="train_lv_loss")
plt.xlabel("epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_df["epoch"], history_df["dice"], label="train_dice")
plt.plot(history_df["epoch"], history_df["iou"], label="train_iou")
plt.plot(history_df["epoch"], history_df["val_dice"], label="val_dice(selected)")
plt.plot(history_df["epoch"], history_df["val_iou"], label="val_iou(selected)")
if "val_dice_student" in history_df.columns:
    plt.plot(history_df["epoch"], history_df["val_dice_student"], linestyle="--", label="val_dice_student")
if "val_dice_ema" in history_df.columns:
    plt.plot(history_df["epoch"], history_df["val_dice_ema"], linestyle="--", label="val_dice_ema")
plt.xlabel("epoch")
plt.legend()
plt.tight_layout()
plt.show()

print("Training started from epoch:", start_epoch)
print("Best checkpoint:", best_path)
print("Best checkpoint epoch:", best_epoch)
print("Best checkpoint model source:", best_model_source)
print("Last checkpoint:", last_path)
print("Saved epoch metrics CSV:", epoch_metrics_csv)
print("Saved epoch metrics JSON:", epoch_metrics_json)

# %% code cell 11
# -----------------------------
# Threshold tuning on validation set
# -----------------------------
best_model = QaTaLViT(vocab_size=len(vocab))
ckpt = torch.load(best_path, map_location=device)
best_model.load_state_dict(ckpt["model"], strict=True)
best_model = best_model.to(device)

best_meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
print("Loaded best checkpoint from:", best_path)
print("best epoch:", ckpt.get("epoch", "n/a"))
print("best_dice in ckpt:", ckpt.get("best_dice", "n/a"))
if isinstance(best_meta, dict) and len(best_meta) > 0:
    print("best checkpoint meta:", best_meta)

best_thr, thr_df = tune_threshold(best_model, val_loader, device=device, thresholds=cfg.threshold_grid)
display(thr_df)
print("Best threshold on val:", best_thr)

best_val_metrics = evaluate(best_model, val_loader, device=device, threshold=best_thr, criterion=criterion)
print("Best val metrics:", best_val_metrics)

# %% code cell 12
# -----------------------------
# Final test with best validation model
# -----------------------------
# best_model is loaded from best_path in Cell 11.
# best_thr is tuned on validation set only.
# Test is evaluated once using that best validation model.
test_metrics = evaluate(best_model, test_loader, device=device, threshold=best_thr, criterion=criterion)
print("Test metrics @ tuned threshold (best validation model):", test_metrics)

# %% code cell 13

# -----------------------------
# Visualize predictions
# -----------------------------
@torch.no_grad()
def visualize_predictions(model, dataset, device, threshold=0.5, n=3):
    """Execute a documented step of the segmentation training pipeline."""
    model.eval()

    idxs = np.linspace(0, len(dataset) - 1, n).astype(int).tolist()
    plt.figure(figsize=(12, 4 * n))

    for row_idx, idx in enumerate(idxs, 1):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        prompt_ids = sample["prompt_ids"].unsqueeze(0).to(device)
        prompt_mask = sample["prompt_mask"].unsqueeze(0).to(device)

        outputs = model(image, prompt_ids, prompt_mask)
        prob = torch.sigmoid(outputs["logits"])[0, 0].cpu().numpy()
        pred = (prob >= threshold).astype(np.float32)
        img = sample["image"][0].cpu().numpy() * TRAIN_STD + TRAIN_MEAN
        gt = sample["mask"][0].cpu().numpy()

        plt.subplot(n, 3, (row_idx - 1) * 3 + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Image | {sample['image_name']}")
        plt.axis("off")

        plt.subplot(n, 3, (row_idx - 1) * 3 + 2)
        plt.imshow(gt, cmap="gray")
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(n, 3, (row_idx - 1) * 3 + 3)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

visualize_predictions(best_model, val_ds, device=device, threshold=best_thr, n=4)

# %% code cell 14
# 15) Prepare full artifacts + Zip full run outputs (run this last)
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader

TASK_NAME_CANONICAL = str(getattr(cfg, 'dataset_hint_name', 'QaTa-Covid19'))
MODEL_NAME = str(getattr(cfg, 'model_name', 'LViT'))
LABEL_PCT = int(round(float(getattr(cfg, 'label_percent', 100.0))))
run_dir = Path(cfg.save_dir)
assert run_dir.exists(), f"Run directory not found: {run_dir}"

models_dir = run_dir / 'models'
visualize_dir = run_dir / 'visualize_val' / '0'
models_dir.mkdir(parents=True, exist_ok=True)
visualize_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Epoch training metrics (CSV + JSON)
# -----------------------------
if 'history' in globals() and len(history) > 0:
    epoch_df = pd.DataFrame(history)
elif (run_dir / 'epoch_metrics.csv').exists():
    epoch_df = pd.read_csv(run_dir / 'epoch_metrics.csv')
else:
    epoch_df = pd.DataFrame()

epoch_csv = run_dir / 'epoch_training_metrics.csv'
epoch_json = run_dir / 'epoch_training_metrics.json'
if len(epoch_df) > 0:
    epoch_df.to_csv(epoch_csv, index=False)
    epoch_json.write_text(epoch_df.to_json(orient='records', indent=2), encoding='utf-8')

# -----------------------------
# 2) Labeled subset metadata
# -----------------------------
subset_meta = {
    'task_name': TASK_NAME_CANONICAL,
    'model_name': MODEL_NAME,
    'label_percent': float(getattr(cfg, 'label_percent', 100.0)),
    'val_selection_source_cfg': str(getattr(cfg, 'val_selection_source', 'best_of_both')),
    'best_epoch': int(globals().get('best_epoch', -1)),
    'best_model_source': str(globals().get('best_model_source', 'unknown')),
    'best_checkpoint_path': str(globals().get('best_path', run_dir / 'qata_lvit_best.pth')),
    'last_checkpoint_path': str(globals().get('last_path', run_dir / 'last_checkpoint.pth')),
    'total_train_samples': int(len(train_df)) if 'train_df' in globals() else 0,
    'labeled_train_samples': int(len(train_labeled_df)) if 'train_labeled_df' in globals() else 0,
    'unlabeled_train_samples': int(len(train_unlabeled_df)) if 'train_unlabeled_df' in globals() else 0,
    'pseudo_enabled': bool(getattr(cfg, 'enable_pseudo_label', False)),
    'pseudo_confidence_threshold': float(getattr(cfg, 'pseudo_confidence_threshold', 0.0)),
    'pseudo_loss_weight': float(getattr(cfg, 'pseudo_loss_weight', 0.0)),
    'pseudo_warmup_epochs': int(getattr(cfg, 'pseudo_warmup_epochs', 0)),
        'epi_beta': float(getattr(cfg, 'epi_beta', 0.99)),
    'seed': int(getattr(cfg, 'seed', 42)),
    'created_at': datetime.now().isoformat(timespec='seconds'),
}
(run_dir / 'labeled_subset_meta.json').write_text(json.dumps(subset_meta, ensure_ascii=False, indent=2), encoding='utf-8')

# -----------------------------
# 3) Metrics report (CSV + JSON)
# -----------------------------
metrics_rows = []

def _safe_float(x, default=np.nan):
    """Execute a documented step of the segmentation training pipeline."""
    try:
        return float(x)
    except Exception:
        return float(default)

if len(epoch_df) > 0:
    last = epoch_df.iloc[-1].to_dict()
    metrics_rows.append({
        'split': 'train',
        'samples': int(len(train_labeled_df)) if 'train_labeled_df' in globals() else 0,
        'loss': _safe_float(last.get('loss', np.nan)),
        'sup_loss': _safe_float(last.get('sup_loss', np.nan)),
        'lv_loss': _safe_float(last.get('lv_loss', np.nan)),
        'dice': _safe_float(last.get('dice', np.nan)),
        'iou': _safe_float(last.get('iou', np.nan)),
    })

if 'best_val_metrics' in globals():
    vm = best_val_metrics
else:
    eval_model = best_model if 'best_model' in globals() else (model if 'model' in globals() else None)
    if eval_model is not None and 'val_loader' in globals():
        vm = evaluate(eval_model, val_loader, device=device, threshold=float(globals().get('best_thr', 0.5)), criterion=criterion)
    else:
        vm = {}

metrics_rows.append({
    'split': 'val',
    'samples': int(len(val_ds)) if 'val_ds' in globals() else 0,
    'loss': _safe_float(vm.get('loss', np.nan)),
    'sup_loss': _safe_float(vm.get('sup_loss', np.nan)),
    'lv_loss': _safe_float(vm.get('lv_loss', np.nan)),
    'dice': _safe_float(vm.get('dice', np.nan)),
    'iou': _safe_float(vm.get('iou', np.nan)),
})

if 'test_metrics' in globals():
    tm = test_metrics
else:
    eval_model = best_model if 'best_model' in globals() else (model if 'model' in globals() else None)
    if eval_model is not None and 'test_loader' in globals():
        tm = evaluate(eval_model, test_loader, device=device, threshold=float(globals().get('best_thr', 0.5)), criterion=criterion)
    else:
        tm = {}

metrics_rows.append({
    'split': 'test',
    'samples': int(len(test_ds)) if 'test_ds' in globals() else 0,
    'loss': _safe_float(tm.get('loss', np.nan)),
    'sup_loss': _safe_float(tm.get('sup_loss', np.nan)),
    'lv_loss': _safe_float(tm.get('lv_loss', np.nan)),
    'dice': _safe_float(tm.get('dice', np.nan)),
    'iou': _safe_float(tm.get('iou', np.nan)),
})

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(run_dir / 'metrics_report.csv', index=False)
(run_dir / 'metrics_report.json').write_text(metrics_df.to_json(orient='records', indent=2), encoding='utf-8')

# -----------------------------
# 4) Pseudo-label vs ground-truth on holdout (unlabeled split)
# -----------------------------
def _pseudo_holdout_report(model_for_eval, dataset, threshold=0.5):
    """Execute a documented step of the segmentation training pipeline."""
    if dataset is None or len(dataset) == 0:
        return {
            'samples': 0,
            'threshold': float(threshold),
            'accuracy': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'dice': np.nan,
            'iou': np.nan,
            'mean_sample_accuracy': np.nan,
            'mean_sample_precision': np.nan,
            'mean_sample_recall': np.nan,
            'mean_sample_dice': np.nan,
            'mean_sample_iou': np.nan,
        }

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    model_for_eval.eval()

    eps = 1e-7
    tp = fp = fn = tn = 0.0
    s_acc = s_pre = s_rec = s_dice = s_iou = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            prompt_ids = batch['prompt_ids'].to(device, non_blocking=True)
            prompt_mask = batch['prompt_mask'].to(device, non_blocking=True)

            out = model_for_eval(images, prompt_ids, prompt_mask)
            prob = torch.sigmoid(out['logits'])
            pred = (prob >= threshold).float()
            gt = (masks > 0.5).float()

            tp_i = float((pred * gt).sum().item())
            fp_i = float((pred * (1 - gt)).sum().item())
            fn_i = float(((1 - pred) * gt).sum().item())
            tn_i = float((((1 - pred) * (1 - gt))).sum().item())

            tp += tp_i; fp += fp_i; fn += fn_i; tn += tn_i

            pre_i = (tp_i + eps) / (tp_i + fp_i + eps)
            rec_i = (tp_i + eps) / (tp_i + fn_i + eps)
            dice_i = (2.0 * tp_i + eps) / (2.0 * tp_i + fp_i + fn_i + eps)
            iou_i = (tp_i + eps) / (tp_i + fp_i + fn_i + eps)
            acc_i = (tp_i + tn_i + eps) / (tp_i + fp_i + fn_i + tn_i + eps)

            s_acc += acc_i; s_pre += pre_i; s_rec += rec_i; s_dice += dice_i; s_iou += iou_i
            n += 1

    report = {
        'samples': int(n),
        'threshold': float(threshold),
        'accuracy': float((tp + tn + eps) / (tp + fp + fn + tn + eps)),
        'precision': float((tp + eps) / (tp + fp + eps)),
        'recall': float((tp + eps) / (tp + fn + eps)),
        'dice': float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)),
        'iou': float((tp + eps) / (tp + fp + fn + eps)),
        'mean_sample_accuracy': float(s_acc / max(1, n)),
        'mean_sample_precision': float(s_pre / max(1, n)),
        'mean_sample_recall': float(s_rec / max(1, n)),
        'mean_sample_dice': float(s_dice / max(1, n)),
        'mean_sample_iou': float(s_iou / max(1, n)),
    }
    return report

eval_model = best_model if 'best_model' in globals() else (model if 'model' in globals() else None)
if eval_model is not None and 'unlabeled_ds' in globals():
    pseudo_report = _pseudo_holdout_report(eval_model, unlabeled_ds, threshold=float(globals().get('best_thr', 0.5)))
else:
    pseudo_report = _pseudo_holdout_report(None, None)

pseudo_df = pd.DataFrame([pseudo_report])
pseudo_df.to_csv(run_dir / 'pseudo_label_vs_ground_truth_holdout.csv', index=False)
(run_dir / 'pseudo_label_vs_ground_truth_holdout.json').write_text(
    json.dumps(pseudo_report, ensure_ascii=False, indent=2),
    encoding='utf-8',
)

# -----------------------------
# 5) Models folder + copy best checkpoint
# -----------------------------
best_ckpt = Path(best_path) if 'best_path' in globals() else (run_dir / 'qata_lvit_best.pth')
if best_ckpt.exists():
    shutil.copy2(best_ckpt, models_dir / best_ckpt.name)

last_ckpt = Path(last_path) if 'last_path' in globals() else (run_dir / 'last_checkpoint.pth')
if last_ckpt.exists():
    shutil.copy2(last_ckpt, models_dir / last_ckpt.name)

# -----------------------------
# 6) Export visualize_val images (pred/gt) for comparison
# -----------------------------
@torch.no_grad()
def _export_visualize_val(model_for_eval, dataset, out_dir, threshold=0.5, max_items=48):
    """Execute a documented step of the segmentation training pipeline."""
    if model_for_eval is None or dataset is None or len(dataset) == 0:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(int(max_items), len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, n).astype(int).tolist()

    model_for_eval.eval()
    saved = 0
    for idx in idxs:
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        mask = sample['mask'][0].cpu().numpy()
        prompt_ids = sample['prompt_ids'].unsqueeze(0).to(device)
        prompt_mask = sample['prompt_mask'].unsqueeze(0).to(device)
        name = str(sample.get('image_name', f'idx_{idx}'))
        stem = Path(name).stem

        out = model_for_eval(image, prompt_ids, prompt_mask)
        prob = torch.sigmoid(out['logits'])[0, 0].cpu().numpy()
        pred = (prob >= threshold).astype(np.uint8) * 255
        gt = (mask >= 0.5).astype(np.uint8) * 255

        cv2.imwrite(str(out_dir / f'{stem}_pred.jpg'), pred)
        cv2.imwrite(str(out_dir / f'{stem}_gt.jpg'), gt)
        saved += 1
    return saved

saved_vis = _export_visualize_val(
    eval_model,
    val_ds if 'val_ds' in globals() else None,
    visualize_dir,
    threshold=float(globals().get('best_thr', 0.5)),
    max_items=48,
)

# -----------------------------
# 7) Compatibility log file name like baseline notebook
# -----------------------------
task_token = TASK_NAME_CANONICAL.replace('-', '_').replace('+', 'Plus')
log_name = f"{task_token}_{MODEL_NAME}_{LABEL_PCT}pct_labels.log"
log_path = run_dir / log_name

log_lines = []
log_lines.append(f"Run dir: {run_dir}")
log_lines.append(f"Created at: {datetime.now().isoformat(timespec='seconds')}")
log_lines.append(f"label_percent={float(getattr(cfg, 'label_percent', 100.0))}")
log_lines.append(f"pseudo_enabled={bool(getattr(cfg, 'enable_pseudo_label', False))}")
log_lines.append(f"epi_beta={float(getattr(cfg, 'epi_beta', 0.99))}")
if len(epoch_df) > 0:
    for _, r in epoch_df.iterrows():
        ep = int(r.get('epoch', 0))
        log_lines.append(
            f"Epoch {ep:03d} loss={float(r.get('loss', np.nan)):.4f} sup_loss={float(r.get('sup_loss', np.nan)):.4f} "
            f"lv_loss={float(r.get('lv_loss', np.nan)):.4f} dice={float(r.get('dice', np.nan)):.4f} iou={float(r.get('iou', np.nan)):.4f}"
        )
log_path.write_text('\n'.join(log_lines), encoding='utf-8')

# -----------------------------
# 8) Zip full run outputs
# -----------------------------
zip_root = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
zip_base = zip_root / f"output_{TASK_NAME_CANONICAL}_{MODEL_NAME}_{LABEL_PCT}pct_labels_FULL"
zip_path = zip_base.with_suffix('.zip')
if zip_path.exists():
    zip_path.unlink()

created = shutil.make_archive(str(zip_base), 'zip', root_dir=str(run_dir.parent), base_dir=run_dir.name)

size_mb = Path(created).stat().st_size / (1024 * 1024)
print('Created full ZIP:', created)
print(f'ZIP size: {size_mb:.2f} MB')
print('Included run dir:', run_dir)
print('Saved visualize_val items:', saved_vis)

must_have = [
    run_dir / 'models',
    run_dir / 'visualize_val',
    run_dir / 'epoch_training_metrics.csv',
    run_dir / 'epoch_training_metrics.json',
    run_dir / 'labeled_subset_meta.json',
    run_dir / 'metrics_report.csv',
    run_dir / 'metrics_report.json',
    run_dir / 'pseudo_label_vs_ground_truth_holdout.csv',
    run_dir / 'pseudo_label_vs_ground_truth_holdout.json',
    log_path,
]
for fp in must_have:
    print(f'{fp.name}:', fp.exists())

# %% [code] cell 5
import os
os.chdir('/kaggle/working')
_run_notebook_shell('python qata_lvit_nobert_gptpro_fixed.py')
