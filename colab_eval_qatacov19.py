"""
Colab script for evaluating LViTImproved on the QaTa-COV19 test set.

Usage in Colab:
1. Put this repository at /content/LViT_improved, or upload this script into the
   repo root.
2. Add kaggle.json at /root/.kaggle/kaggle.json, or set KAGGLE_USERNAME and
   KAGGLE_KEY in the Colab environment.
3. Run:
   !python colab_eval_qatacov19.py

Outputs:
- /content/lvit_qatacov19_eval/test_summary.csv
- /content/lvit_qatacov19_eval/<run_name>/test_metrics.json
- /content/lvit_qatacov19_eval/<run_name>/test_per_sample.csv
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent
WORK_ROOT = Path("/content/lvit_qatacov19_eval")
DATASET_ROOT = WORK_ROOT / "dataset" / "QaTa-Covid19"
KERNEL_OUTPUT_ROOT = WORK_ROOT / "kaggle_outputs"
RESULT_ROOT = WORK_ROOT / "results"

DATASET_REF = "tqc0103/qata-covid19"
RUNS = {
    "lvit-improved-qatacovid19-100pct": {
        "kernel": "tqc0103/lvit-improved-qatacovid19-1-0",
        "subdir": "qatacov19_100pct",
    },
    "lvit-improved-qatacovid19-050pct": {
        "kernel": "tqc0103/lvit-improved-qatacovid19-0-5",
        "subdir": "qatacov19_050pct",
    },
}


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_dependencies() -> None:
    packages = [
        "kaggle==1.6.17",
        "transformers",
        "sentencepiece",
        "protobuf",
        "openpyxl",
        "opencv-python-headless",
        "pandas",
        "tqdm",
    ]
    run_command([sys.executable, "-m", "pip", "install", "-q", *packages])


def ensure_kaggle_auth() -> None:
    kaggle_json = Path("/root/.kaggle/kaggle.json")
    uploaded_json = Path("/content/kaggle.json")
    if uploaded_json.exists() and not kaggle_json.exists():
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(uploaded_json, kaggle_json)
        kaggle_json.chmod(0o600)

    if kaggle_json.exists():
        kaggle_json.chmod(0o600)
        return

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    raise RuntimeError(
        "Kaggle credentials not found. Upload kaggle.json to /content/kaggle.json "
        "or set KAGGLE_USERNAME and KAGGLE_KEY."
    )


def download_inputs() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_marker = DATASET_ROOT / "Test_Folder" / "Test_text.xlsx"
    if not dataset_marker.exists():
        dataset_dir = WORK_ROOT / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        run_command(["kaggle", "datasets", "download", "-d", DATASET_REF, "-p", str(dataset_dir), "--unzip"])

    for run_name, spec in RUNS.items():
        output_dir = KERNEL_OUTPUT_ROOT / run_name
        checkpoint = output_dir / spec["subdir"] / "best_model.pt"
        if checkpoint.exists():
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        run_command(["kaggle", "kernels", "output", spec["kernel"], "-p", str(output_dir)])


def dice_score(target: torch.Tensor, prediction: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target.float().reshape(target.shape[0], -1)
    prediction = prediction.float().reshape(prediction.shape[0], -1)
    intersection = (target * prediction).sum(dim=1)
    denominator = target.sum(dim=1) + prediction.sum(dim=1)
    return (2.0 * intersection + eps) / (denominator + eps)


def iou_score(target: torch.Tensor, prediction: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target.float().reshape(target.shape[0], -1)
    prediction = prediction.float().reshape(prediction.shape[0], -1)
    intersection = (target * prediction).sum(dim=1)
    union = target.sum(dim=1) + prediction.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


def build_test_loader(output_dir: Path, batch_size: int, num_workers: int):
    sys.path.insert(0, str(REPO_ROOT))
    from Load_Dataset import ImageToImage2D, ValGenerator
    from text_encoder import build_cache_metadata
    from utils import read_text

    text_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    max_text_units = 10
    test_text = read_text(str(DATASET_ROOT / "Test_Folder" / "Test_text.xlsx"))
    cache_metadata = build_cache_metadata(model_name=text_model_name, max_units=max_text_units)
    test_dataset = ImageToImage2D(
        str(DATASET_ROOT / "Test_Folder"),
        "QaTa-COV19",
        test_text,
        ValGenerator(output_size=[224, 224]),
        image_size=224,
        cache_dir=str(output_dir / "text_cache" / "Test_Folder"),
        text_model_name=text_model_name,
        local_files_only=False,
        cache_metadata=cache_metadata,
        max_text_units=max_text_units,
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(checkpoint_path: Path, device: torch.device):
    from nets.LViT_improved import LViTImproved
    from text_encoder import attribute_vector_size

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = LViTImproved(
        n_channels=3,
        n_classes=1,
        img_size=224,
        text_dim=int(config.get("text_dim", 768)),
        attribute_dim=attribute_vector_size(),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_one(run_name: str, checkpoint_path: Path, result_dir: Path, batch_size: int, num_workers: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {run_name} on {device}", flush=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    loader = build_test_loader(result_dir, batch_size=batch_size, num_workers=num_workers)
    model, checkpoint = load_model(checkpoint_path, device)
    rows = []
    dice_values = []
    iou_values = []

    with torch.inference_mode():
        for step, (sample, names) in enumerate(loader, start=1):
            image = sample["image"].to(device, non_blocking=True)
            label = sample["label"].to(device, non_blocking=True)
            text = sample["text"].to(device, non_blocking=True)
            attributes = sample.get("attributes")
            if attributes is not None:
                attributes = attributes.to(device, non_blocking=True)

            logits = model(image, text_tokens=text, structured_attributes=attributes)
            prediction = (torch.sigmoid(logits) >= 0.5).float()
            target = (label > 0).float()
            if target.ndim == prediction.ndim - 1:
                target = target.unsqueeze(1)

            batch_dice = dice_score(target, prediction).detach().cpu().tolist()
            batch_iou = iou_score(target, prediction).detach().cpu().tolist()
            dice_values.extend(batch_dice)
            iou_values.extend(batch_iou)
            rows.extend({"image": name, "dice": dice, "iou": iou} for name, dice, iou in zip(names, batch_dice, batch_iou))

            if step % 20 == 0:
                print(f"  {run_name}: processed {len(rows)} samples", flush=True)

    summary = {
        "run": run_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_dice": float(checkpoint.get("best_val_dice", 0.0)),
        "num_test_samples": len(rows),
        "mean_test_dice": float(sum(dice_values) / max(len(dice_values), 1)),
        "mean_test_iou": float(sum(iou_values) / max(len(iou_values), 1)),
    }

    with (result_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (result_dir / "test_per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    ensure_dependencies()
    ensure_kaggle_auth()
    download_inputs()

    batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "16"))
    num_workers = int(os.environ.get("EVAL_NUM_WORKERS", "2"))
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []

    for run_name, spec in RUNS.items():
        checkpoint = KERNEL_OUTPUT_ROOT / run_name / spec["subdir"] / "best_model.pt"
        summaries.append(evaluate_one(run_name, checkpoint, RESULT_ROOT / run_name, batch_size, num_workers))

    summary_csv = WORK_ROOT / "test_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "checkpoint_epoch",
                "checkpoint_best_val_dice",
                "num_test_samples",
                "mean_test_dice",
                "mean_test_iou",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved summary to {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
