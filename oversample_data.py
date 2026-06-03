import os
import shutil

# --- CONFIGURATION ---
BASE_DATASET_DIR = "dataset"
SRC_IMAGES = os.path.join(BASE_DATASET_DIR, "train", "images")
SRC_LABELS = os.path.join(BASE_DATASET_DIR, "train", "labels")

# Your requested multipliers for each class ID
CLASS_MULTIPLIERS = {
    0: 2,   # crazing -> 2x
    1: 10,  # fold -> 10x
    2: 3,   # hole -> 3x
    5: 4,   # pitted_surface -> 4x
    6: 2    # rolled-in_scale -> 2x
}

def oversample_dataset():
    if not os.path.exists(SRC_LABELS):
        print(f"[ERROR] Source labels directory not found: {SRC_LABELS}")
        return

    label_files = [f for f in os.listdir(SRC_LABELS) if f.endswith('.txt')]
    print("[INFO] Starting class-based over-sampling process...")

    copied_images_count = 0

    for label_file in label_files:
        label_path = os.path.join(SRC_LABELS, label_file)

        # Read unique classes present in this specific label file
        file_classes = set()
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if parts:
                    file_classes.add(int(parts[0]))

        # Find the highest multiplier required among the classes in this file
        max_multiplier = 1
        for c_id in file_classes:
            if c_id in CLASS_MULTIPLIERS:
                max_multiplier = max(max_multiplier, CLASS_MULTIPLIERS[c_id])

        # If a multiplier greater than 1 is needed, duplicate the file pair
        if max_multiplier > 1:
            base_name = os.path.splitext(label_file)[0]

            # Find the corresponding image file with correct extension
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                potential_img = base_name + ext
                if os.path.exists(os.path.join(SRC_IMAGES, potential_img)):
                    img_file = potential_img
                    break

            if img_file:
                img_ext = os.path.splitext(img_file)[1]
                src_img_path = os.path.join(SRC_IMAGES, img_file)
                src_lbl_path = os.path.join(SRC_LABELS, label_file)

                # Create copies based on the multiplier (e.g., multiplier 3 means 2 new copies)
                for i in range(1, max_multiplier):
                    new_img_name = f"{base_name}_aug{i}{img_ext}"
                    new_lbl_name = f"{base_name}_aug{i}.txt"

                    dest_img_path = os.path.join(SRC_IMAGES, new_img_name)
                    dest_lbl_path = os.path.join(SRC_LABELS, new_lbl_name)

                    shutil.copy2(src_img_path, dest_img_path)
                    shutil.copy2(src_lbl_path, dest_lbl_path)
                    copied_images_count += 1

    print \
        (f"[SUCCESS] Over-sampling complete! Added {copied_images_count} balanced image-label pairs to the train split.")

if __name__ == "__main__":
    oversample_dataset()