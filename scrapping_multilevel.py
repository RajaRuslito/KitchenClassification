import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from PIL import Image

# Folder utama dataset
BASE_DIR = "dataset_dapur_multilevel"

# Kategori multi-level
queries = {
    "very_clean": ["spotless kitchen", "hygienic kitchen", "sterile kitchen"],
    "clean": ["clean kitchen", "organized kitchen", "neat kitchen"],
    "moderately_clean": ["average clean kitchen", "somewhat clean kitchen"],
    "moderately_dirty": ["messy kitchen", "cluttered kitchen", "kitchen with some dirt"],
    "dirty": ["dirty kitchen", "greasy kitchen", "kitchen with stains"],
    "very_dirty": ["filthy kitchen", "disgusting kitchen", "extremely dirty kitchen"]
}

MAX_NUM = 80  # jumlah gambar per query per engine

def scrape_images():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    for label, keywords in queries.items():
        save_path = os.path.join(BASE_DIR, label)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for keyword in keywords:
            print(f"[INFO] Scraping '{keyword}' into '{label}' folder...")

            # Google
            google_crawler = GoogleImageCrawler(storage={'root_dir': save_path})
            google_crawler.crawl(keyword=keyword, max_num=MAX_NUM)

            # Bing
            bing_crawler = BingImageCrawler(storage={'root_dir': save_path})
            bing_crawler.crawl(keyword=keyword, max_num=MAX_NUM)

    print("\nScraping selesai! Sekarang cek gambar...")

def clean_images():
    deleted = 0
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    os.remove(path)
                    deleted += 1
            except Exception:
                os.remove(path)
                deleted += 1
    print(f"🧹 Cleaning selesai! {deleted} file dihapus.")

if __name__ == "__main__":
    scrape_images()
    clean_images()
    print("🎉 Dataset multi-level dapur siap dipakai!")
