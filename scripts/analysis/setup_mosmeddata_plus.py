from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TEXT_URLS = {
    "train": "https://1drv.ms/x/s!AihndoV8PhTDguIIKCRfYB9Z0NL8Dw?e=8rj6rY",
    "val": "https://1drv.ms/x/c/c3143e7c85766728/QShndoV8PhQggMMGsQAAAAAAtAgZiRQFYfsAjw",
    "test": "https://1drv.ms/x/c/c3143e7c85766728/QShndoV8PhQggMMHsQAAAAAAdHkwXMxGlgU9Tg",
}

SPLITS = {
    "train": ("Train_Folder", "Train_text.xlsx"),
    "val": ("Val_Folder", "Val_text.xlsx"),
    "test": ("Test_Folder", "Test_text.xlsx"),
}


def _download_one(url: str, dest: Path) -> tuple[bool, str]:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    try:
        with urlopen(req, timeout=60) as resp:
            content = resp.read()
        if content[:32].lower().startswith(b"<!doctype html") or b"The request is blocked." in content:
            return False, "OneDrive returned an HTML block page instead of an .xlsx file."
        dest.write_bytes(content)
        return True, f"Downloaded to {dest}"
    except HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except URLError as exc:
        return False, f"URL error: {exc.reason}"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def download_text_annotations(dest_root: Path) -> int:
    dest_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for key, url in TEXT_URLS.items():
        _, filename = SPLITS[key]
        ok, msg = _download_one(url, dest_root / filename)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {filename}: {msg}")
        if not ok:
            failures += 1
    if failures:
        print(
            "\nSome OneDrive links could not be downloaded automatically. "
            "Open the official links in a browser, download the .xlsx files manually, "
            f"and place them under: {dest_root}"
        )
    return failures


def _resolve_text_source(text_root: Path, split_key: str) -> Path:
    _, filename = SPLITS[split_key]
    direct = text_root / filename
    if direct.exists():
        return direct
    alt = text_root / split_key / filename
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Missing {filename} under {text_root}")


def _ensure_split_layout(root: Path, split_dir: str) -> tuple[Path, Path]:
    split_path = root / split_dir
    img_dir = split_path / "img"
    label_dir = split_path / "labelcol"
    if not img_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected prepared split at {split_path} with img/ and labelcol/ subfolders."
        )
    return img_dir, label_dir


def _replace_path(target: Path, source: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    if mode == "symlink":
        target.symlink_to(source, target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def install_dataset(prepared_root: Path, text_root: Path, target_root: Path, mode: str) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for split_key, (split_dir, filename) in SPLITS.items():
        src_img, src_label = _ensure_split_layout(prepared_root, split_dir)
        split_target = target_root / split_dir
        split_target.mkdir(parents=True, exist_ok=True)
        _replace_path(split_target / "img", src_img, mode)
        _replace_path(split_target / "labelcol", src_label, mode)
        text_src = _resolve_text_source(text_root, split_key)
        shutil.copy2(text_src, split_target / filename)

    print(f"Prepared MosMedDataPlus dataset at: {target_root}")
    for split_key, (split_dir, filename) in SPLITS.items():
        split_target = target_root / split_dir
        img_count = sum(1 for p in (split_target / "img").iterdir() if p.is_file())
        mask_count = sum(1 for p in (split_target / "labelcol").iterdir() if p.is_file())
        print(f"- {split_dir}: img={img_count}, labelcol={mask_count}, text={filename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare MosMedDataPlus in the LViT/QaTaLViT folder layout with official text annotations."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download-text", help="Download official MosMedDataPlus text annotation .xlsx files.")
    dl.add_argument(
        "--dest",
        type=Path,
        default=Path("data") / "text_annotations" / "mosmeddata_plus",
        help="Directory to store Train_text.xlsx, Val_text.xlsx, Test_text.xlsx",
    )

    install = sub.add_parser("install", help="Attach text annotations to an already prepared MosMedDataPlus dataset.")
    install.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Existing dataset root that already contains Train_Folder/Val_Folder/Test_Folder with img/ and labelcol/.",
    )
    install.add_argument(
        "--text-root",
        type=Path,
        default=Path("data") / "text_annotations" / "mosmeddata_plus",
        help="Directory that contains Train_text.xlsx, Val_text.xlsx, Test_text.xlsx.",
    )
    install.add_argument(
        "--target-root",
        type=Path,
        default=Path("datasets") / "MosMedDataPlus",
        help="Output dataset root in LViT format.",
    )
    install.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use symlinks to reuse img/labelcol data or copy files physically.",
    )

    combo = sub.add_parser(
        "all",
        help="Try to download text annotations, then install them into an already prepared dataset root.",
    )
    combo.add_argument("--prepared-root", type=Path, required=True)
    combo.add_argument(
        "--text-root",
        type=Path,
        default=Path("data") / "text_annotations" / "mosmeddata_plus",
    )
    combo.add_argument(
        "--target-root",
        type=Path,
        default=Path("datasets") / "MosMedDataPlus",
    )
    combo.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "download-text":
        return download_text_annotations(args.dest)

    if args.cmd == "install":
        install_dataset(args.prepared_root, args.text_root, args.target_root, args.mode)
        return 0

    if args.cmd == "all":
        download_text_annotations(args.text_root)
        install_dataset(args.prepared_root, args.text_root, args.target_root, args.mode)
        return 0

    parser.error(f"Unsupported command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
