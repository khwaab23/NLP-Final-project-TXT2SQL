"""
Download and prepare WikiSQL dataset
"""

from datasets import load_dataset
from pathlib import Path
import json


def download_wikisql(output_dir="./data/wikisql"):
    """Download WikiSQL dataset using Hugging Face datasets"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading WikiSQL dataset...")
    
    try:
        # Load dataset splits
        train_dataset = load_dataset("wikisql", split="train")
        validation_dataset = load_dataset("wikisql", split="validation")
        test_dataset = load_dataset("wikisql", split="test")
        
        # Save to disk
        train_dataset.save_to_disk(str(output_path / "train"))
        validation_dataset.save_to_disk(str(output_path / "validation"))
        test_dataset.save_to_disk(str(output_path / "test"))
        
        print(f"WikiSQL dataset downloaded to {output_dir}")
        
        # Print statistics
        print(f"\nDataset statistics:")
        print(f"  Training examples: {len(train_dataset)}")
        print(f"  Validation examples: {len(validation_dataset)}")
        print(f"  Test examples: {len(test_dataset)}")
        
        # Save metadata
        metadata = {
            "train_size": len(train_dataset),
            "validation_size": len(validation_dataset),
            "test_size": len(test_dataset),
            "source": "wikisql",
            "features": str(train_dataset.features)
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n✅ WikiSQL dataset successfully downloaded!")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download WikiSQL dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/wikisql",
        help="Output directory for dataset"
    )
    
    args = parser.parse_args()
    download_wikisql(args.output_dir)
