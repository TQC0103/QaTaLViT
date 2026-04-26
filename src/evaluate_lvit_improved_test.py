import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Load_Dataset import ImageToImage2D, ValGenerator
from nets.LViT_improved import LViTImproved
from text_encoder import attribute_vector_size, build_cache_metadata
from utils import read_text


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


def load_model(checkpoint_path: Path, device: torch.device, image_size: int) -> tuple[LViTImproved, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    text_dim = int(config.get("text_dim", 768))
    model = LViTImproved(
        n_channels=3,
        n_classes=1,
        img_size=image_size,
        text_dim=text_dim,
        attribute_dim=attribute_vector_size(),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_model_name = args.text_model_name
    cache_metadata = build_cache_metadata(model_name=text_model_name, max_units=args.max_text_units)
    test_text = read_text(str(dataset_root / "Test_Folder" / "Test_text.xlsx"))
    test_dataset = ImageToImage2D(
        str(dataset_root / "Test_Folder"),
        args.task_name,
        test_text,
        ValGenerator(output_size=[args.image_size, args.image_size]),
        image_size=args.image_size,
        cache_dir=str(output_dir / "text_cache" / "Test_Folder"),
        text_model_name=text_model_name,
        local_files_only=args.local_files_only,
        cache_metadata=cache_metadata,
        max_text_units=args.max_text_units,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, checkpoint = load_model(checkpoint_path, device=device, image_size=args.image_size)
    rows = []
    dice_values = []
    iou_values = []

    with torch.inference_mode():
        for sample, names in test_loader:
            image = sample["image"].to(device)
            label = sample["label"].to(device)
            text = sample["text"].to(device)
            attributes = sample.get("attributes")
            if attributes is not None:
                attributes = attributes.to(device)

            logits = model(image, text_tokens=text, structured_attributes=attributes)
            probs = torch.sigmoid(logits)
            pred = (probs >= args.threshold).float()
            target = (label > 0).float()
            if target.ndim == pred.ndim - 1:
                target = target.unsqueeze(1)

            batch_dice = dice_score(target, pred)
            batch_iou = iou_score(target, pred)
            dice_values.extend(batch_dice.detach().cpu().tolist())
            iou_values.extend(batch_iou.detach().cpu().tolist())

            for name, dice, iou in zip(names, batch_dice.detach().cpu().tolist(), batch_iou.detach().cpu().tolist()):
                rows.append({"image": name, "dice": dice, "iou": iou})

    mean_dice = float(sum(dice_values) / max(len(dice_values), 1))
    mean_iou = float(sum(iou_values) / max(len(iou_values), 1))
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_dice": float(checkpoint.get("best_val_dice", 0.0)),
        "dataset_root": str(dataset_root),
        "task_name": args.task_name,
        "split": "Test_Folder",
        "num_samples": len(test_dataset),
        "threshold": args.threshold,
        "device": str(device),
        "mean_test_dice": mean_dice,
        "mean_test_iou": mean_iou,
    }

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_dir / "test_per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-name", default="QaTa-COV19")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--text-model-name",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument("--max-text-units", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
