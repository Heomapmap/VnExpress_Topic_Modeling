import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from pathlib import Path
import pprint

from common_utils import scrape_article, process_datetime, HEADERS

BASE_URL = "https://vnexpress.net"

slugs = ["thoi-su","the-gioi","kinh-doanh","giao-duc","the-thao"
#         "giai-tri","phap-luat","suc-khoe","doi-song","du-lich",
#         "khoa-hoc-cong-nghe","oto-xe-may","y-kien","tam-su",
#         "thu-gian","bat-dong-san"
        ]

CATEGORIES = {s.replace("-", "_"): f"{BASE_URL}/{s}" for s in slugs}

def get_article_links(category_url, limit=100):
    links = []
    page = 1

    while len(links) < limit:
        url = f"{category_url}-p{page}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")
        articles = soup.select("h3.title-news a, h2.title-news a, .title-news a")

        if not articles:
            break

        for a in articles:
            link = a.get("href")
            if link and ("vnexpress.net" in link) and (not link.endswith("#box_comment_vne")):
                links.append(link)

            if len(links) >= limit:
                break

        page += 1
        time.sleep(1)

    return links

if __name__ == "__main__":
    pprint.pprint(CATEGORIES)
    data = []
    LIMIT_PER_CATEGORY = 300

    category_list = list(CATEGORIES.items())
    total_categories = len(category_list)

    for i, (category, category_url) in enumerate(category_list, start=1):
        print(f"\n{'=' * 10} CATEGORY {i}/{total_categories}: {category.upper()} {'=' * 10}")
        links = get_article_links(category_url, limit=LIMIT_PER_CATEGORY)
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

            RAW_DIR = BASE_DIR / "data" / "raw"
            RAW_DIR.mkdir(parents=True, exist_ok=True)

            OUTPUT_FILE = RAW_DIR / "data_web.csv"
            df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig",
                quoting=1,
                escapechar='\\'
            )
            print(f"\nHoàn thành! Đã lưu {len(df)} bài báo tại: {OUTPUT_FILE}")