import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler, FlickrImageCrawler

# Folder utama buat simpan dataset
BASE_DIR = "dataset_dapur"

# Kategori yang mau di-scrap
queries = {
    "clean": ["clean kitchen", "very clean kitchen", "organized kitchen"],
    "dirty": ["dirty kitchen", "messy kitchen", "greasy kitchen", "very dirty kitchen"]
}

# Jumlah gambar per query per engine
MAX_NUM = 100  

def scrape_images():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    for label, keywords in queries.items():
        save_path = os.path.join(BASE_DIR, label)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for keyword in keywords:
            print(f"[INFO] Scraping '{keyword}' into '{label}' folder...")

            # --- Google ---
            google_crawler = GoogleImageCrawler(storage={'root_dir': save_path})
            google_crawler.crawl(keyword=keyword, max_num=MAX_NUM)

            # --- Bing ---
            bing_crawler = BingImageCrawler(storage={'root_dir': save_path})
            bing_crawler.crawl(keyword=keyword, max_num=MAX_NUM)

            # --- Flickr (butuh API Key) ---
            # flickr_crawler = FlickrImageCrawler(
            #     api_key='YOUR_FLICKR_API_KEY',
            #     storage={'root_dir': save_path}
            # )
            # flickr_crawler.crawl(tags=keyword, max_num=MAX_NUM)

    print("\n✅ Dataset scraping selesai!")

if __name__ == "__main__":
    scrape_images()
