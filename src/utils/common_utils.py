import requests
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from underthesea import word_tokenize

CURRENT_DIR = Path(__file__).resolve().parent
STOPWORDS_FILE = CURRENT_DIR / "vn_stopwords.txt"
HEADERS = {"User-Agent": "Mozilla/5.0"}


#Hàm cào dữ liệu bài báo dùng cho 'scrape_***.py'
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


#Hàm xử lý thời gian dùng trong 'scrape_***.py' phục vụ trực quan hóa dữ liệu
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


#Hàm đọc file 'vn_stopwords.txt'
def load_stopwords(path):
    stopwords = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = [p.strip().replace(" ", "_") for p in line.split(",")]
                stopwords.update(p for p in parts if p)
        return stopwords
    else:
        print(f"Cảnh báo: Không tìm thấy file stop words tại {path}")
        return set()

STOPWORDS = load_stopwords(STOPWORDS_FILE)


#Hàm làm sạch text dùng cho 'clean.py' và 'predict_topic.py'
def clean_text(text, return_list=False):
    if not isinstance(text, str):
        return [] if return_list else ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = word_tokenize(text, format="text")
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()

    clean_words = [w for w in words if len(w) > 1
                                and not w.isdigit()
                                and w not in STOPWORDS]

    if return_list:
        return clean_words
    return " ".join(clean_words)


#Tìm và trả về chiếm ưu thế trong 1 Bag of Words dùng cho 'lda_train'
def get_dominant_topic(bow, lda_model):
    topic_dist = lda_model.get_document_topics(bow)
    if not topic_dist:
        return None, 0.0
    dominant_topic = max(topic_dist, key=lambda x: x[1])
    return dominant_topic[0], dominant_topic[1]


#Hàm tính cosine similarity giữa 1 vector và ma trận vector dùng cho 'recommend.py'
def cosine_similarity(vec1,vec2):
    dot_product = np.dot(vec2, vec1)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2, axis=1)
    return dot_product / (norm_v1 * norm_v2)


#Hàm chuyển đổi văn bản thành vector chủ đề dựa trên mô hình LDA dùng cho 'recommend.py'
def get_topic_vector(text, lda_model, dictionary):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(lda_model.num_topics)

    tokens = clean_text(text, return_list=True)
    bow = dictionary.doc2bow(tokens)
    # minimum_probability=0 đảm bảo vector luôn đủ số chiều (số lượng topic)
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    return np.array([prob for _, prob in topic_dist])


#Hàm lọc thời gian dùng cho 'visualize.py'
def filter_by_time(df):
    print("\n--- BỘ LỌC THỜI GIAN ---")
    print("(Nhấn Enter để bỏ qua bộ lọc và xem toàn bộ dữ liệu)")

    year_input = input("Nhập năm muốn xem (VD: 2025): ").strip()
    month_input = input("Nhập tháng muốn xem (1-12): ").strip()

    df_filtered = df.copy()

    df_filtered['publish_date'] = pd.to_datetime(df_filtered['publish_date'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['publish_date'])

    if year_input.isdigit():
        df_filtered = df_filtered[df_filtered['publish_date'].dt.year == int(year_input)]
        print(f"[*] Đã lọc theo năm: {year_input}")

    if month_input.isdigit():
        df_filtered = df_filtered[df_filtered['publish_date'].dt.month == int(month_input)]
        print(f"[*] Đã lọc theo tháng: {month_input}")

    return df_filtered