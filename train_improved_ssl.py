import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from Load_Dataset import ImageToImage2D, RandomGenerator, ValGenerator
from nets.LViT_improved import LViTImproved
from nets.improved_training import ImprovedSSLTrainer
from text_encoder import (
    CachedDomainTextEncoder,
    attribute_vector_size,
    build_cache_metadata,
    save_report_features,
)
from utils import dice_on_batch, read_text


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_history(history: list[dict], save_dir: Path) -> None:
    if not history:
        return
    csv_path = save_dir / "history.csv"
    json_path = save_dir / "history.json"
    fieldnames = list(history[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    save_json(json_path, {"history": history})


def plot_history(history: list[dict], save_dir: Path) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_total"] for entry in history]
    val_dice = [entry["val_dice"] for entry in history]
    lr = [entry["lr"] for entry in history]
    supervised = [entry["train_supervised"] for entry in history]
    consistency = [entry["train_consistency"] for entry in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="train_total")
    axes[0].plot(epochs, supervised, label="train_supervised")
    axes[0].plot(epochs, consistency, label="train_consistency")
    axes[0].set_title("Training Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_dice, label="val_dice")
    axes[1].plot(epochs, lr, label="lr")
    axes[1].set_title("Validation / LR")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(save_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_validation_examples(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    epoch: int,
    max_examples: int = 4,
) -> None:
    if max_examples <= 0:
        return

    example_dir = save_dir / "val_examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            sample, names = batch
            image = sample["image"].to(device)
            label = sample["label"]
            text = sample.get("text")
            attributes = sample.get("attributes")
            if text is not None:
                text = text.to(device)
            if attributes is not None:
                attributes = attributes.to(device)

            logits = model(image, text_tokens=text, structured_attributes=attributes)
            prediction = torch.sigmoid(logits).cpu()

            batch_size = image.shape[0]
            for index in range(batch_size):
                if saved >= max_examples:
                    return
                figure, axes = plt.subplots(1, 3, figsize=(9, 3))
                image_np = image[index].detach().cpu().permute(1, 2, 0).numpy()
                if image_np.shape[-1] == 1:
                    axes[0].imshow(image_np[..., 0], cmap="gray")
                else:
                    axes[0].imshow(image_np)
                axes[0].set_title("Image")
                axes[0].axis("off")

                axes[1].imshow(label[index].cpu().numpy(), cmap="gray")
                axes[1].set_title("GT")
                axes[1].axis("off")

                axes[2].imshow(prediction[index, 0].numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
                axes[2].set_title("Pred")
                axes[2].axis("off")

                figure.tight_layout()
                name = Path(names[index]).stem
                figure.savefig(
                    example_dir / f"epoch_{epoch:03d}_{name}.png",
                    dpi=180,
                    bbox_inches="tight",
                )
                plt.close(figure)
                saved += 1


def checkpoint_payload(
    model: LViTImproved,
    trainer: ImprovedSSLTrainer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_val: float,
    history: list[dict],
    config_payload: dict,
) -> dict:
    return {
        "epoch": epoch,
        "best_val_dice": best_val,
        "model_state": model.state_dict(),
        "teacher_state": trainer.teacher.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "config": config_payload,
    }


def load_checkpoint(
    checkpoint_path: Path,
    model: LViTImproved,
    trainer: ImprovedSSLTrainer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler | None,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    trainer.teacher.load_state_dict(checkpoint["teacher_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    history = checkpoint.get("history", [])
    start_epoch = int(checkpoint.get("epoch", 0))
    best_val = float(checkpoint.get("best_val_dice", 0.0))
    return start_epoch, best_val, history


def precompute_text_cache(
    report_map: dict[str, str],
    cache_dir: str,
    max_text_units: int,
    model_name: str,
    local_files_only: bool = False,
) -> None:
    cache_path = Path(cache_dir)
    metadata = build_cache_metadata(model_name=model_name, max_units=max_text_units)
    missing_items = []
    for sample_id in report_map:
        cache_file = cache_path / f"{Path(sample_id).stem}.pt"
        if not cache_file.exists():
            missing_items.append(sample_id)
            continue
        payload = torch.load(cache_file, map_location="cpu", weights_only=True)
        if payload.get("metadata") != metadata:
            missing_items.append(sample_id)

    if not missing_items:
        return

    encoder = CachedDomainTextEncoder(model_name=model_name, local_files_only=local_files_only)
    for sample_id in missing_items:
        report = report_map[sample_id]
        text, attributes = encoder.encode_report(report, max_lines=max_text_units)
        save_report_features(cache_dir, sample_id, text, attributes, metadata=metadata)


def build_datasets(
    dataset_root: str,
    task_name: str,
    image_size: int,
    text_model_name: str,
    max_text_units: int,
    cache_root: str | None = None,
    local_files_only: bool = False,
):
    train_tf = RandomGenerator(output_size=[image_size, image_size])
    val_tf = ValGenerator(output_size=[image_size, image_size])

    train_text = read_text(f"{dataset_root}/Train_Folder/Train_text.xlsx")
    val_text = read_text(f"{dataset_root}/Val_Folder/Val_text.xlsx")
    if cache_root is None:
        cache_base = Path(dataset_root)
        train_cache = str(cache_base / "Train_Folder" / "text_cache")
        val_cache = str(cache_base / "Val_Folder" / "text_cache")
    else:
        cache_base = Path(cache_root)
        train_cache = str(cache_base / "Train_Folder")
        val_cache = str(cache_base / "Val_Folder")
    cache_metadata = build_cache_metadata(model_name=text_model_name, max_units=max_text_units)
    precompute_text_cache(
        train_text,
        train_cache,
        max_text_units=max_text_units,
        model_name=text_model_name,
        local_files_only=local_files_only,
    )
    precompute_text_cache(
        val_text,
        val_cache,
        max_text_units=max_text_units,
        model_name=text_model_name,
        local_files_only=local_files_only,
    )

    train_dataset = ImageToImage2D(
        f"{dataset_root}/Train_Folder/",
        task_name,
        train_text,
        train_tf,
        image_size=image_size,
        cache_dir=train_cache,
        text_model_name=text_model_name,
        local_files_only=local_files_only,
        cache_metadata=cache_metadata,
        max_text_units=max_text_units,
    )
    val_dataset = ImageToImage2D(
        f"{dataset_root}/Val_Folder/",
        task_name,
        val_text,
        val_tf,
        image_size=image_size,
        cache_dir=val_cache,
        text_model_name=text_model_name,
        local_files_only=local_files_only,
        cache_metadata=cache_metadata,
        max_text_units=max_text_units,
    )
    return train_dataset, val_dataset


def split_labeled_unlabeled(dataset, label_ratio: float, seed: int):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    labeled_count = max(1, int(len(indices) * label_ratio))
    labeled_indices = indices[:labeled_count]
    unlabeled_indices = indices[labeled_count:]
    labeled_subset = Subset(dataset, labeled_indices)
    unlabeled_subset = Subset(dataset, unlabeled_indices) if unlabeled_indices else None
    return labeled_subset, unlabeled_subset


def resolve_subset_names(subset: Subset | None) -> list[str]:
    if subset is None:
        return []
    dataset = subset.dataset
    if hasattr(dataset, "images_list"):
        source_names = dataset.images_list
    elif hasattr(dataset, "mask_list"):
        source_names = dataset.mask_list
    else:
        return [str(index) for index in subset.indices]
    return [str(source_names[index]) for index in subset.indices]


def move_batch_to_device(batch, device: torch.device):
    sample, _ = batch
    moved = {}
    for key, value in sample.items():
        moved[key] = value.to(device)
    return moved


def infinite_loader(loader: Iterable):
    while True:
        yield from loader


def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dices = []
    with torch.no_grad():
        for batch in loader:
            sample = move_batch_to_device(batch, device)
            logits = model(
                sample["image"],
                text_tokens=sample.get("text"),
                structured_attributes=sample.get("attributes"),
            )
            preds = torch.sigmoid(logits)
            dices.append(dice_on_batch(sample["label"], preds))
    return float(sum(dices) / max(len(dices), 1))


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str, epochs: int, min_lr: float):
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--task-name", type=str, default="MoNuSeg")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--label-ratio", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="cosine")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument(
        "--text-model-name",
        type=str,
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument("--max-text-units", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-val-examples", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="./runs/improved_ssl")
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    amp_enabled = bool(args.amp and torch.cuda.is_available())
    effective_batch_size = int(args.batch_size) * int(args.grad_accum_steps)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = build_datasets(
        args.dataset_root,
        args.task_name,
        args.image_size,
        text_model_name=args.text_model_name,
        max_text_units=args.max_text_units,
        cache_root=args.cache_root or str(save_dir / "text_cache"),
        local_files_only=args.local_files_only,
    )
    labeled_dataset, unlabeled_dataset = split_labeled_unlabeled(train_dataset, args.label_ratio, args.seed)
    split_manifest = {
        "task_name": args.task_name,
        "label_ratio": args.label_ratio,
        "seed": args.seed,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "labeled_count": len(labeled_dataset),
        "unlabeled_count": 0 if unlabeled_dataset is None else len(unlabeled_dataset),
        "labeled_samples": resolve_subset_names(labeled_dataset),
        "unlabeled_samples": resolve_subset_names(unlabeled_dataset),
    }
    save_json(save_dir / "split_manifest.json", split_manifest)

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    unlabeled_loader = None
    if unlabeled_dataset is not None and len(unlabeled_dataset) > 0:
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    sample_batch = next(iter(labeled_loader))[0]
    text_dim = sample_batch["text"].shape[-1]
    model = LViTImproved(
        n_channels=sample_batch["image"].shape[1],
        n_classes=1,
        img_size=args.image_size,
        text_dim=text_dim,
        attribute_dim=attribute_vector_size(),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs, args.min_lr)
    trainer = ImprovedSSLTrainer(student=model, optimizer=optimizer, ema_decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    unlabeled_iterator = infinite_loader(unlabeled_loader) if unlabeled_loader is not None else None

    config_payload = {
        **vars(args),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "effective_batch_size": effective_batch_size,
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "labeled_dataset_size": len(labeled_dataset),
        "unlabeled_dataset_size": 0 if unlabeled_dataset is None else len(unlabeled_dataset),
        "text_dim": int(text_dim),
        "trainable_parameters": count_trainable_parameters(model),
    }
    save_json(save_dir / "run_config.json", config_payload)

    history: list[dict] = []
    best_val = 0.0
    start_epoch = 0
    if args.resume:
        start_epoch, best_val, history = load_checkpoint(
            Path(args.resume),
            model=model,
            trainer=trainer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    for epoch in range(start_epoch, args.epochs):
        epoch_metric_lists: dict[str, list[float]] = {
            "total": [],
            "supervised": [],
            "consistency": [],
            "text_guidance": [],
            "boundary": [],
            "prototype": [],
            "fusion_weight": [],
        }

        num_labeled_batches = len(labeled_loader)
        for batch_index, labeled_batch in enumerate(labeled_loader):
            labeled = move_batch_to_device(labeled_batch, device)
            unlabeled = None
            if unlabeled_iterator is not None:
                unlabeled_batch = next(unlabeled_iterator)
                unlabeled = move_batch_to_device(unlabeled_batch, device)

            zero_grad = (batch_index % args.grad_accum_steps) == 0
            step_optimizer = ((batch_index + 1) % args.grad_accum_steps == 0) or ((batch_index + 1) == num_labeled_batches)

            losses = trainer.training_step(
                labeled_images=labeled["image"],
                labeled_masks=labeled["label"].float(),
                labeled_text_tokens=labeled.get("text"),
                labeled_structured_attributes=labeled.get("attributes"),
                unlabeled_images=None if unlabeled is None else unlabeled["image"],
                unlabeled_text_tokens=None if unlabeled is None else unlabeled.get("text"),
                unlabeled_structured_attributes=None if unlabeled is None else unlabeled.get("attributes"),
                scaler=scaler,
                use_amp=amp_enabled,
                grad_accum_steps=args.grad_accum_steps,
                zero_grad=zero_grad,
                step_optimizer=step_optimizer,
            )

            for key in epoch_metric_lists:
                epoch_metric_lists[key].append(float(losses[key].item()))

        if scheduler is not None:
            scheduler.step()

        val_dice = validate(model, val_loader, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": epoch + 1,
            "train_total": sum(epoch_metric_lists["total"]) / max(len(epoch_metric_lists["total"]), 1),
            "train_supervised": sum(epoch_metric_lists["supervised"]) / max(len(epoch_metric_lists["supervised"]), 1),
            "train_consistency": sum(epoch_metric_lists["consistency"]) / max(len(epoch_metric_lists["consistency"]), 1),
            "train_text_guidance": sum(epoch_metric_lists["text_guidance"]) / max(len(epoch_metric_lists["text_guidance"]), 1),
            "train_boundary": sum(epoch_metric_lists["boundary"]) / max(len(epoch_metric_lists["boundary"]), 1),
            "train_prototype": sum(epoch_metric_lists["prototype"]) / max(len(epoch_metric_lists["prototype"]), 1),
            "train_fusion_weight": sum(epoch_metric_lists["fusion_weight"]) / max(len(epoch_metric_lists["fusion_weight"]), 1),
            "val_dice": float(val_dice),
            "lr": current_lr,
        }
        history.append(epoch_record)
        save_history(history, save_dir)
        plot_history(history, save_dir)

        torch.save(
            checkpoint_payload(
                model=model,
                trainer=trainer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                best_val=best_val,
                history=history,
                config_payload=config_payload,
            ),
            save_dir / "last_model.pt",
        )

        is_best = val_dice >= best_val
        if is_best:
            best_val = float(val_dice)
            torch.save(
                checkpoint_payload(
                    model=model,
                    trainer=trainer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    best_val=best_val,
                    history=history,
                    config_payload=config_payload,
                ),
                save_dir / "best_model.pt",
            )
            save_validation_examples(
                model=model,
                loader=val_loader,
                device=device,
                save_dir=save_dir,
                epoch=epoch + 1,
                max_examples=args.save_val_examples,
            )

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(
                checkpoint_payload(
                    model=model,
                    trainer=trainer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    best_val=best_val,
                    history=history,
                    config_payload=config_payload,
                ),
                checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
            )

        run_summary = {
            "best_val_dice": best_val,
            "last_epoch": epoch + 1,
            "effective_batch_size": effective_batch_size,
            "amp_enabled": amp_enabled,
            "device": str(device),
        }
        save_json(save_dir / "run_summary.json", run_summary)

        print(
            f"epoch={epoch + 1} "
            f"train_total={epoch_record['train_total']:.4f} "
            f"train_supervised={epoch_record['train_supervised']:.4f} "
            f"val_dice={val_dice:.4f} "
            f"best_val_dice={best_val:.4f} "
            f"lr={current_lr:.6f}"
        )


if __name__ == "__main__":
    main()
