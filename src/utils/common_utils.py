import requests
from bs4 import BeautifulSoup
import re

HEADERS = {"User-Agent": "Mozilla/5.0"}


def scrape_article(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")

        title_node = soup.select_one("h1.title-detail, h1.title-post, .title_details_news, h1.article-title")
        title = title_node.text.strip() if title_node else ""

        desc_node = soup.select_one("p.description, .description, span.lead, .sapo")
        description = desc_node.text.strip() if desc_node else ""

        content_nodes = soup.select("article.fck_detail p, .fck_detail p, .content_detail p, article p")

        content_parts = []
        for p in content_nodes:
            text = p.text.strip()
            if text and not p.find_parent(class_=["Image", "box-caption"]):
                content_parts.append(text)

        content = "\n".join(content_parts)

        time_node = soup.select_one("span.date, .time-format, .date, .time")
        publish_time = time_node.text.strip() if time_node else ""

        return {
            "title": title,
            "description": description,
            "content": content,
            "publish_time": publish_time,
            "url": url
        }
    except Exception as e:
        print(f"Lỗi tại {url}: {e}")
        return None


def process_datetime(date_str):
    if not date_str: return None, None
    try:
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', date_str)
        time_match = re.search(r'(\d{2}:\d{2})', date_str)

        clean_date = None
        if date_match:
            d, m, y = date_match.group(1).split('/')
            clean_date = f"{y}-{int(m):02d}-{int(d):02d}"

        clean_hour = time_match.group(1) if time_match else None
        return clean_date, clean_hour
    except:
        return None, None