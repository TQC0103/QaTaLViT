import argparse
import subprocess
import sys
from pathlib import Path


def format_ratio_tag(label_ratio: float) -> str:
    return f"{int(round(label_ratio * 100)):03d}pct"


def build_parser(default_save_root: str = "./runs/qatacov19_benchmark") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--save-root", type=str, default=default_save_root)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
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
    parser.add_argument("--resume", action="store_true")
    return parser


def build_train_command(args, ratio: float, save_dir: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "src" / "train_improved_ssl.py"
    command = [
        sys.executable,
        str(script_path),
        "--dataset-root",
        args.dataset_root,
        "--task-name",
        "QaTa-COV19",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--label-ratio",
        str(ratio),
        "--image-size",
        str(args.image_size),
        "--lr",
        str(args.lr),
        "--min-lr",
        str(args.min_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--scheduler",
        args.scheduler,
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--ema-decay",
        str(args.ema_decay),
        "--text-model-name",
        args.text_model_name,
        "--max-text-units",
        str(args.max_text_units),
        "--save-every",
        str(args.save_every),
        "--save-val-examples",
        str(args.save_val_examples),
        "--save-dir",
        str(save_dir),
    ]

    if args.local_files_only:
        command.append("--local-files-only")
    if args.amp:
        command.append("--amp")
    else:
        command.append("--no-amp")
    if args.resume:
        checkpoint_path = save_dir / "last_model.pt"
        if checkpoint_path.exists():
            command.extend(["--resume", str(checkpoint_path)])
    return command


def run_ratio_experiment(args, ratio: float) -> None:
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    ratio_tag = format_ratio_tag(ratio)
    save_dir = save_root / ratio_tag
    command = build_train_command(args, ratio=ratio, save_dir=save_dir)

    print(f"\n=== Running QaTa-COV19 benchmark for {ratio_tag} labels ===")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    for ratio in [1.0, 0.5, 0.25]:
        run_ratio_experiment(args, ratio)


if __name__ == "__main__":
    main()
