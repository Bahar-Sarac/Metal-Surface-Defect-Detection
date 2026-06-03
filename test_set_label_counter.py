import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
# Path to your independent test set labels
LABEL_DIR = "test_set/labels"

# Class names mapping (Exactly matching your data.yaml)
CLASS_MAPPING = {
    0: "crazing",
    1: "fold",
    2: "hole",
    3: "inclusion",
    4: "patches",
    5: "pitted_surface",
    6: "rolled_in_scale",
    7: "scratches"
}

def analyze_labels(label_path):
    class_counts = defaultdict(int)
    total_images = 0

    if not os.path.exists(label_path):
        print(f"Error: The directory '{label_path}' does not exist.")
        return None, 0

    # Scan all .txt files
    for file in os.listdir(label_path):
        if file.endswith(".txt"):
            total_images += 1
            file_path = os.path.join(label_path, file)

            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_id = int(parts[0])
                            if class_id in CLASS_MAPPING:
                                class_counts[class_id] += 1
                        except ValueError:
                            continue

    return class_counts, total_images

def plot_distribution(counts, total_imgs):
    # Prepare data for plotting
    names = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]
    values = [counts[i] for i in range(len(CLASS_MAPPING))]
    total_objects = sum(values)

    # Set style
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 6))

    # Create bar plot
    colors = sns.color_palette("viridis", len(CLASS_MAPPING))
    bars = plt.bar(names, values, color=colors, edgecolor='black', alpha=0.85)

    # Add value tags on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width( ) /2., height + 0.3,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Titles and labels
    plt.title \
        (f"Independent Test Set Class Distribution\n(Total Images: {total_imgs} | Total Bounding Boxes: {total_objects})",
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Defect Class Names", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("Instance Count (Bounding Boxes)", fontsize=12, fontweight='bold', labelpad=10)
    plt.xticks(rotation=15, fontsize=11)
    plt.tight_layout()

    # Save the plot
    output_image = "test_set_distribution.png"
    plt.savefig(output_image, dpi=300)
    print(f"\n📊 Distribution chart successfully saved as '{output_image}'")
    plt.show()

if __name__ == "__main__":
    print("Analyzing test_set structure...")
    counts, total_imgs = analyze_labels(LABEL_DIR)

    if counts is not None:
        total_objects = sum(counts.values())

        # Print a clean terminal report
        print("\n" + "= " *45)
        print(f"{'Class ID':<10}{'Defect Name':<20}{'Instance Count':<15}")
        print("= " *45)
        for cls_id in sorted(CLASS_MAPPING.keys()):
            print(f"{cls_id:<10}{CLASS_MAPPING[cls_id]:<20}{counts[cls_id]:<15}")
        print("= " *45)
        print(f"{'TOTAL':<10}{'Images: ' + str(total_imgs):<20}{'Labels: ' + str(total_objects):<15}")
        print("= " *45)

        # Plot and save
        plot_distribution(counts, total_imgs)