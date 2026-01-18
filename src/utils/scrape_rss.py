import pandas as pd
import feedparser
import time
from pathlib import Path
import pprint

from common_utils import scrape_article, process_datetime

RSS_BASE_URL = "https://vnexpress.net/rss"

slugs = ["thoi-su","the-gioi","kinh-doanh","giao-duc","the-thao",
         "giai-tri","phap-luat","suc-khoe","doi-song","du-lich",
         "khoa-hoc-cong-nghe","oto-xe-may","y-kien","tam-su",
         "thu-gian","bat-dong-san"]

CATEGORIES = {s.replace("-", "_"): f"{RSS_BASE_URL}/{s}.rss" for s in slugs}

def get_article_links_from_rss(rss_url, limit=100):
    print(f"Đang đọc RSS: {rss_url}")
    feed = feedparser.parse(rss_url)
    links = [entry.link for entry in feed.entries[:limit]]
    return links

if __name__ == "__main__":
    pprint.pprint(CATEGORIES)
    data = []
    LIMIT_PER_CATEGORY = 100

    category_list = list(CATEGORIES.items())
    total_categories = len(category_list)

    for i, (category, category_url) in enumerate(category_list, start=1):
        print(f"\n{'=' * 10} CATEGORY {i}/{total_categories}: {category.upper()} {'=' * 10}")
        links = get_article_links_from_rss(category_url, limit=LIMIT_PER_CATEGORY)
        print(f"Found {len(links)} articles")

        for idx, link in enumerate(links, start=1):
            article = scrape_article(link)

            if article:
                date_val, hour_val = process_datetime(article['publish_time'])
                article['publish_date'] = date_val
                article['publish_hour'] = hour_val
                article["category"] = category
                article = {"category": category, **article}
                data.append(article)
                print(f"[{category}] {idx}/{len(links)}: {article['title'][:80]}")

            time.sleep(0.5)

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
            escapechar='\\')

        print(f"\nĐã lưu {len(df)} bài báo vào: {RAW_PATH}")