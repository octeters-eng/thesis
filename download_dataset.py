"""
Download the seed dataset from Google Drive.

Set the environment variable SEED_GDRIVE_FOLDER_ID to the shared folder ID
before running, or export it in your shell profile:

    export SEED_GDRIVE_FOLDER_ID="<your-folder-id>"

Usage:
    python download_dataset.py          # interactive
    python download_dataset.py --yes    # skip re-download prompt
"""

import argparse
import tarfile
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None

from config import GDRIVE_FOLDER_ID, RAW_DATA_DIR, create_directories

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
LABEL_EXTENSIONS = frozenset({".txt", ".xml", ".json"})
_SEPARATOR = "=" * 60


def download_from_gdrive(folder_id: str, output_dir: Path) -> bool:
    """Download an entire Google Drive folder using gdown.

    Args:
        folder_id:  The Google Drive folder ID (not the full URL).
        output_dir: Local directory to download files into.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    if gdown is None:
        print("ERROR: gdown is not installed. Run: pip install -r requirements.txt")
        return False

    if not folder_id:
        print("ERROR: Google Drive folder ID is not set.")
        print("Export SEED_GDRIVE_FOLDER_ID before running this script.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading dataset from Google Drive ...")
    print(f"  URL : {url}")
    print(f"  Dest: {output_dir}")

    try:
        gdown.download_folder(url=url, output=str(output_dir), quiet=False, use_cookies=False)
        print("Download complete.")
        return True
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("\nTo download manually:")
        print(f"  1. Open: {url}")
        print(f"  2. Download all files and place them in: {output_dir}")
        return False


def extract_archives(data_dir: Path) -> None:
    """Extract any .zip or .tar* archives found recursively in *data_dir*.

    Args:
        data_dir: Directory to search for archives.
    """
    for archive in data_dir.rglob("*.zip"):
        print(f"Extracting {archive.name} ...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(archive.parent)

    for archive in data_dir.rglob("*.tar*"):
        print(f"Extracting {archive.name} ...")
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(archive.parent)


def show_dataset_summary(data_dir: Path) -> None:
    """Print a summary of the downloaded dataset directory.

    Args:
        data_dir: Root of the downloaded dataset.
    """
    print("\n" + _SEPARATOR)
    print("DATASET SUMMARY")
    print(_SEPARATOR)

    images = [f for f in data_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    labels = [f for f in data_dir.rglob("*") if f.is_file() and f.suffix.lower() in LABEL_EXTENSIONS]

    print(f"Total image files: {len(images)}")
    print(f"Total label files: {len(labels)}")

    print("\nDirectory structure:")
    for item in sorted(data_dir.rglob("*")):
        if item.is_dir():
            depth  = len(item.relative_to(data_dir).parts)
            indent = "  " * depth
            print(f"{indent}📁 {item.name}/")

    # File-type breakdown
    ext_counts: dict[str, int] = {}
    for f in data_dir.rglob("*"):
        if f.is_file():
            ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1

    if ext_counts:
        print("\nFile types:")
        for ext, count in sorted(ext_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {ext or '(no ext)'}: {count}")

    # Look for a class definition file
    class_file_names = ["classes.txt", "classes.names", "obj.names", "data.yaml", "dataset.yaml"]
    classes_file = next(
        (found[0] for name in class_file_names if (found := list(data_dir.rglob(name)))),
        None,
    )
    if classes_file:
        print(f"\nClasses file: {classes_file}")
        print(classes_file.read_text())
    else:
        print("\nNo classes file found.")


def main(force_yes: bool = False) -> None:
    create_directories()

    print(_SEPARATOR)
    print("SEED RECOGNITION DATASET DOWNLOADER")
    print(_SEPARATOR)

    existing_files = list(RAW_DATA_DIR.rglob("*"))
    if any(f.is_file() for f in existing_files):
        print(f"\nData already exists in {RAW_DATA_DIR}")
        if force_yes:
            redownload = True
        else:
            redownload = input("Re-download? (y/n): ").strip().lower() == "y"
        if not redownload:
            show_dataset_summary(RAW_DATA_DIR)
            return

    if download_from_gdrive(GDRIVE_FOLDER_ID, RAW_DATA_DIR):
        extract_archives(RAW_DATA_DIR)
        show_dataset_summary(RAW_DATA_DIR)

    print("\nNext step: python prepare_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download seed dataset from Google Drive.")
    parser.add_argument("--yes", action="store_true", help="Skip re-download confirmation prompt.")
    args = parser.parse_args()
    main(force_yes=args.yes)
