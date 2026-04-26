from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"C:\Users\ASUS\OneDrive\Documents\GitHub\LViT_improved")
RUN1 = ROOT / ".tmp_metrics" / "100_1"
RUN2 = ROOT / ".tmp_metrics" / "100_res"
OUTDIR = ROOT / "report" / "figures" / "qata100_analysis"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    epoch1 = pd.read_csv(RUN1 / "epoch_metrics.csv")
    epoch2 = pd.read_csv(RUN2 / "epoch_metrics.csv")
    combined = (
        pd.concat([epoch1, epoch2], ignore_index=True)
        .sort_values("epoch")
        .drop_duplicates(subset=["epoch"], keep="last")
        .reset_index(drop=True)
    )

    summary1 = pd.read_csv(RUN1 / "metrics_report.csv")
    summary2 = pd.read_csv(RUN2 / "metrics_report.csv")
    train_mask = pd.read_csv(RUN1 / "train_mask_area.csv")
    val_mask = pd.read_csv(RUN1 / "val_mask_area.csv")
    return combined, summary1, summary2, train_mask, val_mask


def save_training_curves(df: pd.DataFrame) -> None:
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

    axes[1, 1].plot(df["epoch"], df["val_dice_student"] * 100, label="Student val Dice", color="#17becf", linewidth=2)
    axes[1, 1].plot(df["epoch"], df["val_dice_ema"] * 100, label="EMA val Dice", color="#e377c2", linewidth=2)
    axes[1, 1].set_title("Validation Dice: student vs EMA")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Dice (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("QaTa-COV19 100% labels: combined training curves (epochs 1-60)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qata100_combined_curves.png", bbox_inches="tight")
    plt.close(fig)


def save_summary_bars(summary1: pd.DataFrame, summary2: pd.DataFrame, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    labels = ["Train", "Val", "Test"]
    x = range(len(labels))
    dice1 = (summary1["dice"] * 100).tolist()
    dice2 = (summary2["dice"] * 100).tolist()
    iou1 = (summary1["iou"] * 100).tolist()
    iou2 = (summary2["iou"] * 100).tolist()

    width = 0.35
    axes[0].bar([i - width / 2 for i in x], dice1, width=width, label="Epoch 30 summary", color="#4c78a8")
    axes[0].bar([i + width / 2 for i in x], dice2, width=width, label="Epoch 60 summary", color="#f58518")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("Dice (%)")
    axes[0].set_title("Dice by split")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([i - width / 2 for i in x], iou1, width=width, label="Epoch 30 summary", color="#54a24b")
    axes[1].bar([i + width / 2 for i in x], iou2, width=width, label="Epoch 60 summary", color="#b279a2")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylabel("mIoU (%)")
    axes[1].set_title("mIoU by split")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("QaTa-COV19 100% labels: split-level summary comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qata100_summary_bars.png", bbox_inches="tight")
    plt.close(fig)

    best_idx = df["val_dice"].idxmax()
    best_row = df.loc[best_idx]
    final_row = summary2.set_index("split")
    summary = {
        "best_val_epoch": int(best_row["epoch"]),
        "best_val_dice": float(best_row["val_dice"]),
        "best_val_iou": float(best_row["val_iou"]),
        "final_epoch": int(df["epoch"].max()),
        "final_train_dice": float(final_row.loc["train", "dice"]),
        "final_val_dice": float(final_row.loc["val", "dice"]),
        "final_test_dice": float(final_row.loc["test", "dice"]),
        "final_train_iou": float(final_row.loc["train", "iou"]),
        "final_val_iou": float(final_row.loc["val", "iou"]),
        "final_test_iou": float(final_row.loc["test", "iou"]),
    }
    (OUTDIR / "qata100_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_mask_area_plots(train_mask: pd.DataFrame, val_mask: pd.DataFrame) -> None:
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

    fig.suptitle("QaTa-COV19 mask area distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "qata100_mask_area.png", bbox_inches="tight")
    plt.close(fig)


def save_epoch_tables(df: pd.DataFrame) -> None:
    selected_epochs = [1, 5, 10, 20, 30, 40, 43, 50, 60]
    snap = df[df["epoch"].isin(selected_epochs)][
        ["epoch", "loss", "dice", "iou", "val_loss", "val_dice", "val_iou"]
    ].copy()
    snap[["dice", "iou", "val_dice", "val_iou"]] *= 100
    snap.to_csv(OUTDIR / "qata100_epoch_snapshots.csv", index=False)
    df.to_csv(OUTDIR / "qata100_combined_epoch_metrics.csv", index=False)


def save_gap_plot(df: pd.DataFrame) -> None:
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
    fig.savefig(OUTDIR / "qata100_generalization_gap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    combined, summary1, summary2, train_mask, val_mask = load_data()
    save_training_curves(combined)
    save_summary_bars(summary1, summary2, combined)
    save_mask_area_plots(train_mask, val_mask)
    save_epoch_tables(combined)
    save_gap_plot(combined)
    print(f"Saved outputs to: {OUTDIR}")


if __name__ == "__main__":
    main()
