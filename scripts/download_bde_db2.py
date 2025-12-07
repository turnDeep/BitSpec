#!/usr/bin/env python3
"""
BDE-db2 Dataset Download Script

Downloads the BDE-db2 dataset (531,244 BDEs) from GitHub and Figshare
for BonDNet retraining.

Dataset Information:
    - Paper: "Expansion of bond dissociation prediction with machine learning
             to medicinally and environmentally relevant chemical space"
             Digital Discovery (RSC), 2023
    - GitHub: https://github.com/patonlab/BDE-db2
    - Figshare: https://figshare.com/articles/dataset/bde-db2_csv_gz/19367051
    - Size: 531,244 unique BDEs from 65,540 molecules
    - Elements: C, H, N, O, S, Cl, F, P, Br, I

Usage:
    python scripts/download_bde_db2.py --output data/external/bde-db2
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import gzip
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_file(url, output_path, description="Downloading"):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

    logger.info(f"Downloaded: {output_path}")


def clone_github_repo(output_dir):
    """Clone BDE-db2 GitHub repository with Git LFS support"""
    repo_url = "https://github.com/patonlab/BDE-db2.git"
    repo_dir = output_dir / "BDE-db2-repo"

    if repo_dir.exists():
        logger.info(f"Repository already exists: {repo_dir}")
        logger.info("Pulling latest changes...")
        subprocess.run(["git", "pull"], cwd=repo_dir, check=True)
    else:
        logger.info(f"Cloning BDE-db2 repository from {repo_url}")
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
        logger.info(f"Repository cloned: {repo_dir}")

    # Pull Git LFS files if LFS is used
    logger.info("Checking for Git LFS files...")
    try:
        # Check if git-lfs is installed
        result = subprocess.run(
            ["git", "lfs", "version"],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("Git LFS detected, pulling large files...")
            subprocess.run(["git", "lfs", "pull"], cwd=repo_dir, check=True)
            logger.info("Git LFS files downloaded")
        else:
            logger.warning("Git LFS not installed, skipping LFS pull")
    except FileNotFoundError:
        logger.warning("Git LFS not available, large files may not be downloaded")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git LFS pull failed: {e}")

    return repo_dir


def download_figshare_dataset(output_dir, repo_dir=None):
    """Download BDE-db2 dataset from GitHub repository or Figshare"""
    output_file = output_dir / "bde-db2.csv.gz"

    if output_file.exists():
        logger.info(f"Dataset already exists: {output_file}")
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        return output_file

    # Try to copy from cloned GitHub repository first (preferred method)
    if repo_dir and repo_dir.exists():
        repo_dataset = repo_dir / "Datasets" / "bde-db2" / "bde-db2.csv.gz"
        if repo_dataset.exists():
            logger.info(f"Copying dataset from GitHub repository: {repo_dataset}")
            shutil.copy2(repo_dataset, output_file)
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"Copied: {output_file} ({file_size_mb:.2f} MB)")
            return output_file
        else:
            logger.warning(f"Dataset not found in repository: {repo_dataset}")

    # Fallback: Try Figshare download (may not work due to API changes)
    logger.info("Dataset not found in repository, trying Figshare...")
    logger.info("Note: Figshare direct download may not work. If it fails,")
    logger.info("please ensure the GitHub repository was cloned successfully.")

    figshare_url = "https://figshare.com/ndownloader/files/34266542"

    try:
        download_file(figshare_url, output_file, "Downloading bde-db2.csv.gz")
    except Exception as e:
        logger.error(f"Failed to download from Figshare: {e}")
        logger.info("Please manually download from:")
        logger.info("https://figshare.com/articles/dataset/bde-db2_csv_gz/19367051")
        logger.info(f"And place the file at: {output_file}")
        logger.info("Or ensure the GitHub repository is cloned with:")
        logger.info(f"  git clone https://github.com/patonlab/BDE-db2.git {repo_dir}")
        return None

    return output_file


def extract_and_validate(compressed_file, output_dir):
    """Extract compressed dataset and validate"""
    if not compressed_file or not compressed_file.exists():
        logger.error("Compressed file not found")
        return False

    extracted_file = output_dir / "bde-db2.csv"

    if extracted_file.exists():
        logger.info(f"Extracted file already exists: {extracted_file}")
    else:
        logger.info(f"Extracting {compressed_file.name}...")
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(extracted_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"Extracted: {extracted_file}")

    # Validate file
    logger.info("Validating dataset...")
    line_count = 0
    with open(extracted_file, 'r') as f:
        header = f.readline()
        for line in f:
            line_count += 1

    logger.info(f"Dataset contains {line_count:,} entries")
    logger.info(f"Header: {header.strip()}")

    expected_entries = 531244
    if abs(line_count - expected_entries) > 1000:
        logger.warning(f"Expected ~{expected_entries:,} entries, found {line_count:,}")
        logger.warning("Dataset may be incomplete")
    else:
        logger.info("✓ Dataset validation successful")

    return True


def create_readme(output_dir, repo_dir):
    """Create README with dataset information"""
    readme_content = f"""# BDE-db2 Dataset

Downloaded from: https://github.com/patonlab/BDE-db2

## Dataset Information

**Paper:** "Expansion of bond dissociation prediction with machine learning
to medicinally and environmentally relevant chemical space"
- Published: Digital Discovery (RSC), 2023
- DOI: 10.1039/D3DD00169E

**Dataset Statistics:**
- Total BDEs: 531,244 unique bond dissociation energies
- Molecules: 65,540
- Elements: C, H, N, O, S, Cl, F, P, Br, I (10 elements)
- Calculation Level: M06-2X/def2-TZVP (DFT)
- Properties: Enthalpies and Gibbs free energies

**Data Source:**
- PubChem compound library
- ZINC15 database (38,277 small molecules added)

## Files

- `bde-db2.csv.gz` - Compressed dataset (downloaded from Figshare)
- `bde-db2.csv` - Extracted dataset
- `BDE-db2-repo/` - GitHub repository clone (models, code, examples)

## Usage

Convert to BonDNet format:
```bash
python scripts/convert_bde_db2_to_bondnet.py \\
  --input {output_dir}/bde-db2.csv \\
  --output data/processed/bondnet_training/
```

## Citation

If you use this dataset, please cite:
```
@article{{sowndarya2023expansion,
  title={{Expansion of bond dissociation prediction with machine learning
         to medicinally and environmentally relevant chemical space}},
  author={{Sowndarya, S. V. Shree and Kim, Yeonjoon and Kim, Seonah and
          St. John, Peter C. and Paton, Robert S.}},
  journal={{Digital Discovery}},
  year={{2023}},
  publisher={{Royal Society of Chemistry}}
}}
```

## Download Date

{Path(__file__).stat().st_mtime}

## Repository Structure

```
{output_dir}/
├── bde-db2.csv.gz          # Compressed dataset
├── bde-db2.csv             # Extracted dataset
├── BDE-db2-repo/           # GitHub repository
│   ├── Datasets/           # Additional dataset files
│   ├── Models/             # Pre-trained GNN models
│   └── Example-BDE-prediction/  # Jupyter notebooks
└── README.md               # This file
```
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    logger.info(f"Created README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download BDE-db2 dataset for BonDNet retraining"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/external/bde-db2',
        help='Output directory for dataset (default: data/external/bde-db2)'
    )
    parser.add_argument(
        '--skip-repo',
        action='store_true',
        help='Skip cloning GitHub repository (only download dataset)'
    )
    parser.add_argument(
        '--skip-extract',
        action='store_true',
        help='Skip extraction (keep compressed format only)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BDE-db2 Dataset Download Script")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("")

    # Step 1: Clone GitHub repository (optional)
    repo_dir = None
    if not args.skip_repo:
        try:
            repo_dir = clone_github_repo(output_dir)
            logger.info("✓ GitHub repository cloned successfully")
        except Exception as e:
            logger.warning(f"Failed to clone repository: {e}")
            logger.warning("Continuing with dataset download only...")

    # Step 2: Download dataset from GitHub repository or Figshare
    compressed_file = download_figshare_dataset(output_dir, repo_dir)

    if not compressed_file:
        logger.error("Failed to download dataset")
        logger.info("\nManual download instructions:")
        logger.info("1. Visit: https://figshare.com/articles/dataset/bde-db2_csv_gz/19367051")
        logger.info("2. Download 'bde-db2.csv.gz'")
        logger.info(f"3. Place it in: {output_dir}")
        logger.info("4. Re-run this script")
        sys.exit(1)

    logger.info("✓ Dataset downloaded successfully")

    # Step 3: Extract and validate
    if not args.skip_extract:
        if extract_and_validate(compressed_file, output_dir):
            logger.info("✓ Dataset extracted and validated")
        else:
            logger.error("Dataset validation failed")
            sys.exit(1)

    # Step 4: Create README
    create_readme(output_dir, repo_dir)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Download Complete!")
    logger.info("=" * 60)
    logger.info(f"Dataset location: {output_dir.absolute()}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Convert dataset to BonDNet format:")
    logger.info(f"   python scripts/convert_bde_db2_to_bondnet.py \\")
    logger.info(f"     --input {output_dir}/bde-db2.csv \\")
    logger.info(f"     --output data/processed/bondnet_training/")
    logger.info("")
    logger.info("2. Retrain BonDNet:")
    logger.info("   See: docs/BONDNET_RETRAINING.md")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
