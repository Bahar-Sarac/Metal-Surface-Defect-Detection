import os
from ultralytics import YOLO

def main():
    # 1. Define Paths
    # Ensure data.yaml path is correct relative to this script
    yaml_path = os.path.join("dataset", "data.yaml")

    # 2. Load YOLO Nano Model Structure
    # Using standard nano configuration as the baseline for custom training
    print("[INFO] Initializing YOLO Nano model configuration...")
    model = YOLO("yolov26n.yaml")  # Modern custom training syntax uses the architectural definition

    # 3. Start Professional Custom Training
    print("[INFO] Starting model training with advanced industrial augmentations...")

    results = model.train(
        data=yaml_path,
        epochs=150,  # Adjusted for industrial surface defect convergence
        imgsz=640,  # Standard resolution for robust feature detection
        batch=16,  # Safe batch size for standard GPU memory (VRAM) bounds
        device=0,  # Uses GPU 0. Change to 'cpu' if no NVIDIA GPU is available
        workers=4,  # Multi-threaded data loading
        save=True,  # Automatically saves best and last weights (.pt)
        project="Metal_Defect_Project",
        name="version_1",

        # --- YOUR ADVANCED AUGMENTATION HYPERPARAMETERS ---
        fliplr=0.5,  # Horizontal flip probability (50% chance)
        flipud=0.5,  # Vertical flip probability (50% chance)
        degrees=90.0,  # Rotate image by +/- 90 degrees randomly
        hsv_v=0.20,  # Brightness (Value) adjustment (+/- 20%)
        hsv_s=0.15,  # Contrast (Saturation) adjustment (+/- 15%)
        blur=0.01,  # Gaussian blur probability (Simulates movement blur)
        mosaic=1.0,  # Mosaic augmentation active (Blends 4 images into 1)

        # --- OPTIMIZATION ---
        optimizer="SGD",  # Stable optimizer for heavy class-imbalance scenarios
        lr0=0.01,  # Initial learning rate
    )

    print("[SUCCESS] Training process completed successfully.")


if __name__ == "__main__":
    main()