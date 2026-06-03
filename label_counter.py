import os
from collections import Counter
import matplotlib.pyplot as plt

# 1. KULLANICI AYARLARI (KLASÖR YOLU)
# Proje klasörünün içindeki veri seti klasör adını tam olarak buraya yazın:
BASE_DATASET_DIR = "dataset"

# YOLO data.yaml dosyanızdaki sınıf indeksleri ve isimleri:
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

# 2. VERİ SETİNİ TARAMA FONKSİYONU
def get_class_counts_from_split(split_name):
    """Belirtilen split (train, valid, test) içindeki etiketleri sayar."""
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

# 3. ANA ANALİZ SÜRECİ
def main():
    if not os.path.exists(BASE_DATASET_DIR):
        print(f"Hata: '{BASE_DATASET_DIR}' klasörü bulunamadı! Lütfen klasör adını kontrol edin.")
        return

    # Her bir küme için sayım yapalım
    train_counts = get_class_counts_from_split("train")
    val_counts = get_class_counts_from_split("valid")
    test_counts = get_class_counts_from_split("test")

    # Tüm sınıfların benzersiz ID listesini alalım (0'dan 7'ye kadar)
    all_class_ids = sorted(list(CLASS_NAMES.keys()))

    # Toplam istatistikleri hesaplamak ve tablo oluşturmak için listeler
    names = [CLASS_NAMES[c_id] for c_id in all_class_ids]
    total_counts = []

    # 4. TERMİNALE DETAYLI TABLO YAZDIRMA
    print("\n" + "=" * 75)
    print(f" 📊 VERİ SETİ DETAYLI SINIF DAĞILIM RAPORU ({BASE_DATASET_DIR})")
    print("=" * 75)
    print(f"{'Sınıf ID':<10}{'Kusur Adı':<20}{'Train':<10}{'Valid':<10}{'Test':<10}{'TOPLAM':<10}")
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

    # 5. MATPLOTLIB İLE TOPLAM GRAFİĞİ ÇİZDİRME
    plt.figure(figsize=(12, 6))

    # Genel toplam barlarını çizelim
    bars = plt.bar(names, total_counts, color='#4a90e2', edgecolor='black', alpha=0.85)

    # Barların üzerine net toplam adetlerini yazalım
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(total_counts) * 0.01),
                 f'{yval}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel('Kusur Sınıfları', fontsize=12, fontweight='bold')
    plt.ylabel('Toplam Nesne (Etiket) Sayısı', fontsize=12, fontweight='bold')
    plt.title('Tüm Veri Setindeki Toplam Sınıf Dağılımı (Train + Val + Test)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    print("\n[INFO] Toplam dağılım grafiği ekrana getiriliyor...")
    plt.show()


if __name__ == "__main__":
    main()