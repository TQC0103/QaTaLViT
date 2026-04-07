import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


BUNDLE_NAME = "lvit_improved_kaggle_bundle"
ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
BUNDLE_DIR = DIST_DIR / BUNDLE_NAME
ZIP_PATH = DIST_DIR / f"{BUNDLE_NAME}.zip"

FILES_TO_COPY = [
    ".gitignore",
    "KAGGLE_TRAIN.md",
    "KAGGLE_NOTEBOOK_CELLS.md",
    "requirements-kaggle.txt",
    "Load_Dataset.py",
    "text_encoder.py",
    "utils.py",
    "train_improved_ssl.py",
    "run_qatacov19_benchmark.py",
    "run_qatacov19_100pct.py",
    "run_qatacov19_050pct.py",
    "run_qatacov19_025pct.py",
    "nets/__init__.py",
    "nets/LViT_improved.py",
    "nets/improved_ssl.py",
    "nets/improved_training.py",
]


def copy_item(relative_path: str) -> None:
    source = ROOT / relative_path
    destination = BUNDLE_DIR / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_manifest() -> None:
    manifest_path = BUNDLE_DIR / "bundle_manifest.txt"
    manifest_lines = ["Kaggle bundle contents:"]
    manifest_lines.extend(FILES_TO_COPY)
    manifest_lines.append("")
    manifest_lines.append("Upload this folder as your Kaggle code dataset.")
    manifest_lines.append("If you upload the zip instead, unzip it inside the notebook first.")
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED) as archive:
        for path in BUNDLE_DIR.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(BUNDLE_DIR))


def main() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    for relative_path in FILES_TO_COPY:
        copy_item(relative_path)

    write_manifest()
    make_zip()
    print(f"Bundle directory: {BUNDLE_DIR}")
    print(f"Bundle zip: {ZIP_PATH}")


if __name__ == "__main__":
    main()
