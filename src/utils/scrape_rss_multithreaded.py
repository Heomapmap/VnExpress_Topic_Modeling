import pandas as pd
import feedparser
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from common_utils import scrape_article, process_datetime

RSS_BASE_URL = "https://vnexpress.net/rss"

slugs = [
    "thoi-su","the-gioi","kinh-doanh","giao-duc","the-thao",
    "giai-tri","phap-luat","suc-khoe","doi-song","du-lich",
    "khoa-hoc-cong-nghe","oto-xe-may","y-kien","tam-su",
    "thu-gian","bat-dong-san"
]

CATEGORIES = {s.replace("-", "_"): f"{RSS_BASE_URL}/{s}.rss" for s in slugs}


def get_article_links_from_rss(rss_url, limit=100):
    print(f"Đang đọc RSS: {rss_url}")
    feed = feedparser.parse(rss_url)
    return [entry.link for entry in feed.entries[:limit]]


def process_article(category, link):
    article = scrape_article(link)
    if article:
        date_val, hour_val = process_datetime(article['publish_time'])
        article.update({
            'publish_date' : date_val,
            'publish_hour' : hour_val,
            'category' : category
        })
        return article
    else:
        print(f"[ERROR] {category}: {link}")
    return None


if __name__ == "__main__":
    data = []
    LIMIT_PER_CATEGORY = 3
    MAX_WORKERS = 10

    category_list = list(CATEGORIES.items())
    total_categories = len(category_list)

    for i, (category, category_url) in enumerate(CATEGORIES.items()):
        print(f"\n{'=' * 10} CATEGORY {i}/{total_categories}: {category.upper()} {'=' * 10}")
        links = get_article_links_from_rss(category_url, limit=LIMIT_PER_CATEGORY)
        print(f"Found {len(links)} articles")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_article, category, link) for link in links]
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result:
                    data.append(result)
                    print(f"[{category}] {idx}/{len(links)}: {result['title'][:80]}")

        time.sleep(3)


    if data:
        df = pd.DataFrame(data)

        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        RAW_PATH = BASE_DIR / "data" / "raw" / "data_rss.csv"
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            RAW_PATH,
            index=False,
            encoding="utf-8-sig",
            quoting=1,
            escapechar='\\'
        )

        print(f"\nĐã lưu {len(df)} bài báo vào: {RAW_PATH}")
    else:
        print("\nKhông có dữ liệu để lưu.")