import sys
from pathlib import Path

# Set up paths
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "tmp" / "StyleTTS2"))

import meldataset

# Read train_list.txt
train_list_path = project_root / "kokoro_vietnamese" / "data" / "train_list.txt"
with open(train_list_path, "r", encoding="utf-8") as f:
    data_list = f.readlines()

print(f"Loaded {len(data_list)} lines from train_list.txt")

# Initialize dataset
dataset = meldataset.FilePathDataset(
    data_list=data_list,
    root_path=str(project_root / "kokoro_vietnamese"),
    OOD_data=str(project_root / "data" / "OOD_texts.txt"),
    min_length=5
)

print("Dataset initialized successfully!")
print("Attempting to get item 0...")
item = dataset[0]
print("Successfully got item 0!")
print("Item 0 elements:")
print(f"Speaker ID: {item[0]}")
print(f"Acoustic Feature (Mel) shape: {item[1].shape}")
print(f"Text Tensor shape: {item[2].shape}")
print(f"Ref Text shape: {item[3].shape}")
print(f"Ref Mel shape: {item[4].shape}")
print(f"Ref Label: {item[5]}")
print(f"Path: {item[6]}")
print(f"Wave shape: {item[7].shape}")
print("ALL TESTS PASSED!")
