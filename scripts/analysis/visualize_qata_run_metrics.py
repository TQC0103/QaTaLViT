from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\ASUS\OneDrive\Documents\GitHub\LViT_improved")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize merged QaTa-COV19 run metrics.")
    parser.add_argument("--run1", required=True, type=Path, help="Directory for the first run chunk.")
    parser.add_argument("--run2", required=True, type=Path, help="Directory for the second run chunk.")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory to save figures and csv/json files.")
    parser.add_argument("--tag", required=True, help="Short tag for output filenames, e.g. qata50.")
    parser.add_argument("--title", required=True, help="Human-readable title for plots.")
    return parser.parse_args()


def load_run_bundle(run1: Path, run2: Path) -> dict[str, pd.DataFrame]:
    epoch1 = pd.read_csv(run1 / "epoch_metrics.csv")
    epoch2 = pd.read_csv(run2 / "epoch_metrics.csv")
    combined = (
        pd.concat([epoch1, epoch2], ignore_index=True)
        .sort_values("epoch")
        .drop_duplicates(subset=["epoch"], keep="last")
        .reset_index(drop=True)
    )

    bundle = {
        "combined": combined,
        "summary1": pd.read_csv(run1 / "metrics_report.csv"),
        "summary2": pd.read_csv(run2 / "metrics_report.csv"),
        "train_mask": pd.read_csv(run1 / "train_mask_area.csv"),
        "val_mask": pd.read_csv(run1 / "val_mask_area.csv"),
        "pseudo1": pd.read_csv(run1 / "pseudo_label_vs_ground_truth_holdout.csv"),
        "pseudo2": pd.read_csv(run2 / "pseudo_label_vs_ground_truth_holdout.csv"),
    }
    return bundle


def save_training_curves(df: pd.DataFrame, outdir: Path, tag: str, title: str) -> None:
    best_idx = df["val_dice"].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val_dice = float(df.loc[best_idx, "val_dice"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)

    axes[0, 0].plot(df["epoch"], df["loss"], label="Train loss", color="#1f77b4", linewidth=2)
    axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val loss", color="#d62728", linewidth=2)
    axes[0, 0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Loss curve")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(df["epoch"], df["dice"] * 100, label="Train Dice", color="#2ca02c", linewidth=2)
    axes[0, 1].plot(df["epoch"], df["val_dice"] * 100, label="Val Dice", color="#ff7f0e", linewidth=2)
    axes[0, 1].scatter([best_epoch], [best_val_dice * 100], color="black", s=30, zorder=5)
    axes[0, 1].annotate(
        f"Best val Dice\nE{best_epoch}: {best_val_dice*100:.2f}%",
        (best_epoch, best_val_dice * 100),
        xytext=(8, -22),
        textcoords="offset points",
        fontsize=8,
    )
    axes[0, 1].set_title("Dice curve")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(df["epoch"], df["iou"] * 100, label="Train mIoU", color="#9467bd", linewidth=2)
    axes[1, 0].plot(df["epoch"], df["val_iou"] * 100, label="Val mIoU", color="#8c564b", linewidth=2)
    axes[1, 0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_title("mIoU curve")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("mIoU (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    if {"val_dice_student", "val_dice_ema"}.issubset(df.columns):
        axes[1, 1].plot(df["epoch"], df["val_dice_student"] * 100, label="Student val Dice", color="#17becf", linewidth=2)
        axes[1, 1].plot(df["epoch"], df["val_dice_ema"] * 100, label="EMA val Dice", color="#e377c2", linewidth=2)
        axes[1, 1].set_title("Validation Dice: student vs EMA")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Dice (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.25)
    else:
        axes[1, 1].axis("off")

    fig.suptitle(f"{title}: combined training curves (epochs 1-60)", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_combined_curves.png", bbox_inches="tight")
    plt.close(fig)


def save_summary_bars(summary1: pd.DataFrame, summary2: pd.DataFrame, outdir: Path, tag: str, title: str) -> dict[str, float]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    labels = ["Train", "Val", "Test"]
    x = range(len(labels))
    dice1 = (summary1["dice"] * 100).tolist()
    dice2 = (summary2["dice"] * 100).tolist()
    iou1 = (summary1["iou"] * 100).tolist()
    iou2 = (summary2["iou"] * 100).tolist()

    width = 0.35
    axes[0].bar([i - width / 2 for i in x], dice1, width=width, label="Chunk 1 summary", color="#4c78a8")
    axes[0].bar([i + width / 2 for i in x], dice2, width=width, label="Chunk 2 summary", color="#f58518")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("Dice (%)")
    axes[0].set_title("Dice by split")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([i - width / 2 for i in x], iou1, width=width, label="Chunk 1 summary", color="#54a24b")
    axes[1].bar([i + width / 2 for i in x], iou2, width=width, label="Chunk 2 summary", color="#b279a2")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylabel("mIoU (%)")
    axes[1].set_title("mIoU by split")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(f"{title}: split-level summary comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_summary_bars.png", bbox_inches="tight")
    plt.close(fig)

    final_row = summary2.set_index("split")
    return {
        "final_train_dice": float(final_row.loc["train", "dice"]),
        "final_val_dice": float(final_row.loc["val", "dice"]),
        "final_test_dice": float(final_row.loc["test", "dice"]),
        "final_train_iou": float(final_row.loc["train", "iou"]),
        "final_val_iou": float(final_row.loc["val", "iou"]),
        "final_test_iou": float(final_row.loc["test", "iou"]),
    }


def save_mask_area_plots(train_mask: pd.DataFrame, val_mask: pd.DataFrame, outdir: Path, tag: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    axes[0].hist(train_mask["mask_area"], bins=50, alpha=0.75, label="Train", color="#1f77b4")
    axes[0].hist(val_mask["mask_area"], bins=50, alpha=0.65, label="Val", color="#ff7f0e")
    axes[0].set_title("Mask area histogram")
    axes[0].set_xlabel("Mask area (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    train_sorted = train_mask["mask_area"].sort_values().reset_index(drop=True)
    val_sorted = val_mask["mask_area"].sort_values().reset_index(drop=True)
    train_cdf = (train_sorted.index + 1) / len(train_sorted)
    val_cdf = (val_sorted.index + 1) / len(val_sorted)

    axes[1].plot(train_sorted, train_cdf, label="Train CDF", color="#1f77b4", linewidth=2)
    axes[1].plot(val_sorted, val_cdf, label="Val CDF", color="#ff7f0e", linewidth=2)
    axes[1].set_title("Mask area empirical CDF")
    axes[1].set_xlabel("Mask area (pixels)")
    axes[1].set_ylabel("Cumulative ratio")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.suptitle(f"{title}: mask area distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_mask_area.png", bbox_inches="tight")
    plt.close(fig)


def save_epoch_tables(df: pd.DataFrame, outdir: Path, tag: str) -> tuple[int, float, float]:
    best_idx = df["val_dice"].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val_dice = float(df.loc[best_idx, "val_dice"])
    best_val_iou = float(df.loc[best_idx, "val_iou"])

    selected_epochs = sorted({1, 5, 10, 20, 30, 40, best_epoch, 50, 60})
    snap = df[df["epoch"].isin(selected_epochs)][
        ["epoch", "loss", "dice", "iou", "val_loss", "val_dice", "val_iou"]
    ].copy()
    snap[["dice", "iou", "val_dice", "val_iou"]] *= 100
    snap.to_csv(outdir / f"{tag}_epoch_snapshots.csv", index=False)
    df.to_csv(outdir / f"{tag}_combined_epoch_metrics.csv", index=False)
    return best_epoch, best_val_dice, best_val_iou


def save_gap_plot(df: pd.DataFrame, outdir: Path, tag: str, title: str) -> None:
    gap_dice = (df["dice"] - df["val_dice"]) * 100
    gap_iou = (df["iou"] - df["val_iou"]) * 100

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    ax.plot(df["epoch"], gap_dice, label="Train-Val Dice gap", color="#d62728", linewidth=2)
    ax.plot(df["epoch"], gap_iou, label="Train-Val mIoU gap", color="#1f77b4", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Generalization gap across epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap (percentage points)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_generalization_gap.png", bbox_inches="tight")
    plt.close(fig)


def save_semi_supervised_dynamics(df: pd.DataFrame, outdir: Path, tag: str, title: str) -> None:
    columns = {"train_pseudo_valid_ratio", "epi_bank_size"}
    if not columns.issubset(df.columns):
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    axes[0].plot(df["epoch"], df["train_pseudo_valid_ratio"] * 100, color="#ff7f0e", linewidth=2)
    axes[0].set_title("Pseudo-label valid ratio")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Valid pseudo-label ratio (%)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["epoch"], df["epi_bank_size"], color="#2ca02c", linewidth=2)
    axes[1].set_title("EPI memory bank size")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Number of entries")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"{title}: semi-supervised dynamics", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_semi_supervised_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def save_pseudo_quality_plot(pseudo1: pd.DataFrame, pseudo2: pd.DataFrame, outdir: Path, tag: str, title: str) -> dict[str, float]:
    if pseudo1.empty or pseudo2.empty:
        return {}

    metrics = [
        ("dice", "Global Dice"),
        ("iou", "Global mIoU"),
        ("mean_sample_dice", "Mean-sample Dice"),
        ("mean_sample_iou", "Mean-sample mIoU"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    x = range(len(metrics))
    width = 0.35
    vals1 = [float(pseudo1.iloc[0][col]) * 100 for col, _ in metrics]
    vals2 = [float(pseudo2.iloc[0][col]) * 100 for col, _ in metrics]
    labels = [label for _, label in metrics]

    axes[0].bar([i - width / 2 for i in x], vals1, width=width, label=f"Chunk 1 (thr={float(pseudo1.iloc[0]['threshold']):.2f})", color="#4c78a8")
    axes[0].bar([i + width / 2 for i in x], vals2, width=width, label=f"Chunk 2 (thr={float(pseudo2.iloc[0]['threshold']):.2f})", color="#f58518")
    axes[0].set_xticks(list(x), labels, rotation=15, ha="right")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Pseudo-label holdout quality")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    sample_metrics = [
        ("precision", "Global precision"),
        ("recall", "Global recall"),
        ("mean_sample_precision", "Mean precision"),
        ("mean_sample_recall", "Mean recall"),
    ]
    vals1_b = [float(pseudo1.iloc[0][col]) * 100 for col, _ in sample_metrics]
    vals2_b = [float(pseudo2.iloc[0][col]) * 100 for col, _ in sample_metrics]
    labels_b = [label for _, label in sample_metrics]

    x2 = range(len(sample_metrics))
    axes[1].bar([i - width / 2 for i in x2], vals1_b, width=width, label="Chunk 1", color="#54a24b")
    axes[1].bar([i + width / 2 for i in x2], vals2_b, width=width, label="Chunk 2", color="#b279a2")
    axes[1].set_xticks(list(x2), labels_b, rotation=15, ha="right")
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Pseudo-label precision/recall")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(f"{title}: pseudo-label holdout comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / f"{tag}_pseudo_quality.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "pseudo_threshold_chunk1": float(pseudo1.iloc[0]["threshold"]),
        "pseudo_threshold_chunk2": float(pseudo2.iloc[0]["threshold"]),
        "pseudo_mean_sample_dice_chunk1": float(pseudo1.iloc[0]["mean_sample_dice"]),
        "pseudo_mean_sample_dice_chunk2": float(pseudo2.iloc[0]["mean_sample_dice"]),
        "pseudo_mean_sample_iou_chunk1": float(pseudo1.iloc[0]["mean_sample_iou"]),
        "pseudo_mean_sample_iou_chunk2": float(pseudo2.iloc[0]["mean_sample_iou"]),
    }


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = load_run_bundle(args.run1, args.run2)
    combined = bundle["combined"]

    save_training_curves(combined, outdir, args.tag, args.title)
    summary_stats = save_summary_bars(bundle["summary1"], bundle["summary2"], outdir, args.tag, args.title)
    save_mask_area_plots(bundle["train_mask"], bundle["val_mask"], outdir, args.tag, args.title)
    best_epoch, best_val_dice, best_val_iou = save_epoch_tables(combined, outdir, args.tag)
    save_gap_plot(combined, outdir, args.tag, args.title)
    save_semi_supervised_dynamics(combined, outdir, args.tag, args.title)
    pseudo_stats = save_pseudo_quality_plot(bundle["pseudo1"], bundle["pseudo2"], outdir, args.tag, args.title)

    summary = {
        "best_val_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "final_epoch": int(combined["epoch"].max()),
        **summary_stats,
        **pseudo_stats,
    }
    (outdir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
