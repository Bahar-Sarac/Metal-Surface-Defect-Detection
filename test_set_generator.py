import os
import shutil
from collections import defaultdict

# --- CONFIGURATION ---
# Root directory of your main dataset
DATASET_ROOT = "dataset"
# Subfolder names for the dataset splits to scan
SPLITS = ["train", "valid", "test"]

# Target directory structure for the independent test set
TARGET_ROOT = "test_set"
TARGET_IMAGES = os.path.join(TARGET_ROOT, "images")
TARGET_LABELS = os.path.join(TARGET_ROOT, "labels")

# Target configuration
NUM_CLASSES = 8
TARGET_COUNT_PER_CLASS = 20  # Updated to 20 instances per class

# Create target directories
os.makedirs(TARGET_IMAGES, exist_ok=True)
os.makedirs(TARGET_LABELS, exist_ok=True)

# 1. MAP ALL LABEL FILES AND THEIR CONTAINED CLASSES (COUNTING EACH INSTANCE)
# Structure: { file_name: [class_id_1, class_id_2, class_id_2, ...] }
file_annotations = {}
# Structure: { file_name: (source_image_path, source_label_path) }
file_paths = {}

for split in SPLITS:
    img_dir = os.path.join(DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(DATASET_ROOT, split, "labels")

    if not os.path.exists(lbl_dir) or not os.path.exists(img_dir):
        continue

    for label_file in os.listdir(lbl_dir):
        if not label_file.endswith(".txt"):
            continue

        base_name = os.path.splitext(label_file)[0]

        # Find the matching image extension (jpg, jpeg, png)
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            if os.path.exists(os.path.join(img_dir, base_name + ext)):
                img_file = base_name + ext
                break

        if not img_file:
            continue  # Skip if no matching image is found

        label_path = os.path.join(lbl_dir, label_file)
        image_path = os.path.join(img_dir, img_file)

        # Read label file and extract EVERY class instance line by line
        classes_in_file = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                        if 0 <= cls_id < NUM_CLASSES:
                            classes_in_file.append(cls_id)  # Adds every single instance
                    except ValueError:
                        continue

        if classes_in_file:
            file_annotations[base_name] = classes_in_file
            file_paths[base_name] = (image_path, label_path)

# 2. BALANCED EXTRACTION ALGORITHM WITH INSTANCE COUNTING
collected_counts = defaultdict(int)
moved_files = set()

print("Starting balanced data migration (Target: 20 instances per class)...")

# Iterate through each class to collect the target amount
for target_cls in range(NUM_CLASSES):
    if collected_counts[target_cls] >= TARGET_COUNT_PER_CLASS:
        continue

    # Filter files that contain the target class and haven't been moved yet
    candidate_files = [
        f for f, cls_list in file_annotations.items()
        if target_cls in cls_list and f not in moved_files
    ]

    # Sort candidates by total number of instances contained (fewer total instances first)
    # This ensures fine-grained control over the distribution balance
    candidate_files.sort(key=lambda f: len(file_annotations[f]))

    for f_name in candidate_files:
        # Check if we already reached the target count for this specific class due to co-occurrence
        if collected_counts[target_cls] >= TARGET_COUNT_PER_CLASS:
            break

        src_img, src_lbl = file_paths[f_name]
        dst_img = os.path.join(TARGET_IMAGES, os.path.basename(src_img))
        dst_lbl = os.path.join(TARGET_LABELS, os.path.basename(src_lbl))

        # MOVE the files safely (using shutil.move)
        shutil.move(src_img, dst_img)
        shutil.move(src_lbl, dst_lbl)

        moved_files.add(f_name)

        # Crucial Step: Count EVERY single instance inside the moved file
        for cls in file_annotations[f_name]:
            collected_counts[cls] += 1

# Class names mapping for the final report
class_names = {
    0: "crazing", 1: "fold", 2: "hole", 3: "inclusion",
    4: "patches", 5: "pitted_surface", 6: "rolled_in_scale", 7: "scratches"
}

# 3. PRINT SUMMARY REPORT
print("\n--- Process Completed! ---")
print(f"Total image-label pairs successfully moved: {len(moved_files)}")
print("\nFinal exact instance counts in 'test_set':")
for cls_id in range(NUM_CLASSES):
    print(f"Class {cls_id} ({class_names[cls_id]}): {collected_counts[cls_id]} instances.")