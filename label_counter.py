import os
from collections import Counter
import matplotlib.pyplot as plt

# ==========================================
# 1. USER SETTINGS (DATASET PATH & CLASSES)
# ==========================================
# Specify the exact dataset directory name inside your project folder:
BASE_DATASET_DIR = "dataset"

# Class indices and names matching your YOLO data.yaml file:
CLASS_NAMES = {
    0: "crazing",
    1: "fold",
    2: "hole",
    3: "inclusion",
    4: "patches",
    5: "pitted_surface",
    6: "rolled-in_scale",
    7: "scratches"
}

# ==========================================
# 2. DATASET SCANNING FUNCTION
# ==========================================
def get_class_counts_from_split(split_name):
    """
    Counts the object labels inside the specified data split (train, valid, test).
    """
    labels_path = os.path.join(BASE_DATASET_DIR, split_name, "labels")
    counts = Counter()

    if not os.path.exists(labels_path):
        return counts

    txt_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    for file_name in txt_files:
        file_path = os.path.join(labels_path, file_name)
        with open(file_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    counts[class_id] += 1
    return counts

# ==========================================
# 3. MAIN ANALYSIS PROCESS
# ==========================================
def main():
    if not os.path.exists(BASE_DATASET_DIR):
        print(f"Error: Directory '{BASE_DATASET_DIR}' not found! Please check the folder path.")
        return

    # Count objects for each data split
    train_counts = get_class_counts_from_split("train")
    val_counts = get_class_counts_from_split("valid")
    test_counts = get_class_counts_from_split("test")

    # Get the sorted list of unique class IDs (0 to 7)
    all_class_ids = sorted(list(CLASS_NAMES.keys()))

    # Lists to store aggregated data for plotting and reporting
    names = [CLASS_NAMES[c_id] for c_id in all_class_ids]
    total_counts = []

    # ------------------------------------------
    # 4. PRINT DETAILED TABLE TO TERMINAL
    # ------------------------------------------
    print("\n" + "=" * 75)
    print(f" 📊 DETAILED DATASET CLASS DISTRIBUTION REPORT ({BASE_DATASET_DIR})")
    print("=" * 75)
    print(f"{'Class ID':<10}{'Defect Name':<20}{'Train':<10}{'Valid':<10}{'Test':<10}{'TOTAL':<10}")
    print("-" * 75)

    for c_id in all_class_ids:
        tr = train_counts[c_id]
        vl = val_counts[c_id]
        ts = test_counts[c_id]
        total = tr + vl + ts
        total_counts.append(total)

        name = CLASS_NAMES[c_id]
        print(f"{c_id:<10}{name:<20}{tr:<10}{vl:<10}{ts:<10}{total:<10}")

    print("-" * 75)
    print(
        f"{'TOTAL':<30}{sum(train_counts.values()):<10}{sum(val_counts.values()):<10}{sum(test_counts.values()):<10}{sum(total_counts):<10}")
    print("=" * 75)

    # ------------------------------------------
    # 5. PLOT TOTAL DISTRIBUTION WITH MATPLOTLIB
    # ------------------------------------------
    plt.figure(figsize=(12, 6))

    # Plot the total count bars
    bars = plt.bar(names, total_counts, color='#4a90e2', edgecolor='black', alpha=0.85)

    # Display the exact total numbers on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(total_counts) * 0.01),
                 f'{yval}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel('Defect Classes', fontsize=12, fontweight='bold')
    plt.ylabel('Total Object (Label) Count', fontsize=12, fontweight='bold')
    plt.title('Total Class Distribution Across Entire Dataset (Train + Val + Test)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    print("\n[INFO] Displaying the total distribution graph...")
    plt.show()


if __name__ == "__main__":
    main()