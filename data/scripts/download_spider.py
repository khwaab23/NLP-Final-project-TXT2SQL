"""
Download and prepare Spider dataset
"""

import os
import zipfile
import urllib.request
from pathlib import Path
import json


def download_spider(output_dir="./data/spider"):
    """Download Spider dataset from official source"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Spider dataset URL
    SPIDER_URL = "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"
    
    zip_path = output_path / "spider.zip"
    
    print("Downloading Spider dataset...")
    print("Note: You might need to download manually from:")
    print("https://yale-lily.github.io/spider")
    
    try:
        urllib.request.urlretrieve(SPIDER_URL, zip_path)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        
        print(f"Spider dataset downloaded and extracted to {output_dir}")
        
        # Clean up zip file
        zip_path.unlink()
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease download Spider dataset manually:")
        print("1. Visit: https://yale-lily.github.io/spider")
        print("2. Download the dataset")
        print(f"3. Extract to: {output_dir}")
    
    # Verify dataset
    verify_spider(output_path)


def verify_spider(data_path: Path):
    """Verify Spider dataset is properly downloaded"""
    
    required_files = [
        "train_spider.json",
        "train_others.json",
        "dev.json",
        "tables.json"
    ]
    
    print("\nVerifying dataset...")
    missing_files = []
    
    for file in required_files:
        file_path = data_path / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"✓ Found: {file}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Please ensure Spider dataset is properly downloaded.")
    else:
        print("\n✅ All required files found!")
        
        # Print statistics
        with open(data_path / "train_spider.json", 'r') as f:
            train_data = json.load(f)
        with open(data_path / "dev.json", 'r') as f:
            dev_data = json.load(f)
        
        print(f"\nDataset statistics:")
        print(f"  Training examples: {len(train_data)}")
        print(f"  Development examples: {len(dev_data)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Spider dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/spider",
        help="Output directory for dataset"
    )
    
    args = parser.parse_args()
    download_spider(args.output_dir)
