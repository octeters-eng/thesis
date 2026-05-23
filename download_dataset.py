"""
Download dataset from Google Drive.
Usage: python download_dataset.py
"""

import argparse
import zipfile
import tarfile
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None

from config import GDRIVE_FOLDER_ID, RAW_DATA_DIR, create_directories


def download_from_gdrive(folder_id: str, output_dir: Path):
    """Download entire folder from Google Drive."""
    if gdown is None:
        print("ERROR: gdown is not installed. Run: pip install -r requirements.txt")
        return False

    if not folder_id:
        print("ERROR: Google Drive folder ID is empty.")
        print("Set environment variable SEED_GDRIVE_FOLDER_ID before running download.")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading dataset from Google Drive...")
    print(f"Folder URL: {url}")
    print(f"Output directory: {output_dir}")
    
    try:
        gdown.download_folder(
            url=url,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
        )
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error downloading with gdown: {e}")
        print("\nIf automatic download fails, please:")
        print(f"  1. Go to: {url}")
        print(f"  2. Download all files manually")
        print(f"  3. Place them in: {output_dir}")
        return False
    
    return True


def extract_archives(data_dir: Path):
    """Extract any zip/tar archives found in the data directory."""
    for archive in data_dir.rglob("*.zip"):
        print(f"Extracting {archive.name}...")
        with zipfile.ZipFile(archive, 'r') as zf:
            zf.extractall(archive.parent)
        print(f"  Extracted to {archive.parent}")
    
    for archive in data_dir.rglob("*.tar*"):
        print(f"Extracting {archive.name}...")
        with tarfile.open(archive, 'r:*') as tf:
            tf.extractall(archive.parent)
        print(f"  Extracted to {archive.parent}")


def show_dataset_summary(data_dir: Path):
    """Show summary of downloaded dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    label_extensions = {'.txt', '.xml', '.json'}
    
    images = []
    labels = []
    
    for f in data_dir.rglob("*"):
        if f.is_file():
            if f.suffix.lower() in image_extensions:
                images.append(f)
            elif f.suffix.lower() in label_extensions:
                labels.append(f)
    
    print(f"Total image files: {len(images)}")
    print(f"Total label files: {len(labels)}")
    
    # Show directory structure
    print("\nDirectory structure:")
    for item in sorted(data_dir.rglob("*")):
        if item.is_dir():
            depth = len(item.relative_to(data_dir).parts)
            indent = "  " * depth
            print(f"{indent}📁 {item.name}/")
    
    # Show file type distribution
    ext_counts = {}
    for f in data_dir.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    
    if ext_counts:
        print("\nFile types:")
        for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
            print(f"  {ext}: {count} files")
    
    # Check for class information
    classes_file = None
    for name in ["classes.txt", "classes.names", "obj.names", "data.yaml", "dataset.yaml"]:
        found = list(data_dir.rglob(name))
        if found:
            classes_file = found[0]
            break
    
    if classes_file:
        print(f"\nClasses file found: {classes_file}")
        print("Contents:")
        with open(classes_file, 'r') as f:
            content = f.read()
            print(content)
    else:
        print("\nNo classes file found. You may need to create one.")
    
    return images, labels


def main(force_yes: bool = False):
    create_directories()
    
    print("=" * 60)
    print("SEED RECOGNITION DATASET DOWNLOADER")
    print("=" * 60)
    
    # Check if data already exists
    existing_files = list(RAW_DATA_DIR.rglob("*"))
    if any(f.is_file() for f in existing_files):
        print(f"\nData already exists in {RAW_DATA_DIR}")
        if force_yes:
            should_redownload = True
        else:
            response = input("Re-download? (y/n): ").strip().lower()
            should_redownload = response == "y"
        if not should_redownload:
            show_dataset_summary(RAW_DATA_DIR)
            return
    
    # Download
    success = download_from_gdrive(GDRIVE_FOLDER_ID, RAW_DATA_DIR)
    
    if success:
        extract_archives(RAW_DATA_DIR)
        show_dataset_summary(RAW_DATA_DIR)
    
    print("\nNext step: Run 'python prepare_dataset.py' to prepare data for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download seed dataset from Google Drive")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Assume yes for re-download prompt.",
    )
    args = parser.parse_args()
    main(force_yes=args.yes)
