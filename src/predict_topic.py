import gensim
import csv
import os
from pathlib import Path
import textwrap

from utils.common_utils import scrape_article, clean_text
from utils.summarize import get_ai_insight

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lda" / "lda_model.model"
DICT_PATH = BASE_DIR / "models" / "lda" / "id2word.dictionary"
RESULT_FILE = BASE_DIR / "data" / "processed" / "results.csv"

class TopicPredictor:
    def __init__(self):
        print("\n--- Đang tải Model & Dictionary ---")
        self.dictionary = gensim.corpora.Dictionary.load(str(DICT_PATH))
        self.lda_model = gensim.models.LdaModel.load(str(MODEL_PATH))

    def run_pipeline(self, url):
        article = scrape_article(url)
        if not article or not article['content']:
            return None, (None, 0)

        raw_text = f"{article['title']} {article['description']} {article['content']}"
        tokens = clean_text(raw_text, return_list=True)
        bow = self.dictionary.doc2bow(tokens)
        topic_dist = self.lda_model.get_document_topics(bow)

        if topic_dist:
            best_topic = max(topic_dist, key=lambda x: x[1])
            return article, best_topic
        return article, (None, 0)

if __name__ == "__main__":
    predictor = TopicPredictor()

    url_input = input("\nNhập link VnExpress: ")

    article_data, result = predictor.run_pipeline(url_input)

    if article_data and result[0] is not None:
        topic_id, score = result

        print("--- Đang phân tích nội dung chuyên sâu bằng AI... ---")
        summary, highlights = get_ai_insight(
            article_data['title'],
            article_data['description'],
            article_data['content']
        )

        print(f"\n" + "=" * 50)
        print(f"Tiêu đề: {article_data['title']}")
        print(f"Chủ đề (LDA): Topic {topic_id} ({score:.2%})")
        wrapped_summary = textwrap.fill(summary, width=80, initial_indent="    ", subsequent_indent="    ")
        print(f"Tóm tắt: \n{wrapped_summary}")
        indented_highlights = highlights.replace("\n", "\n\t")
        print(f"Ý chính: \n\t{indented_highlights}")
        print("=" * 50)

        RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

        new_record = {
            "url": url_input,
            "title": article_data['title'],
            "topic_id": topic_id,
            "confidence": f"{score:.2%}",
            "summary": summary,
            "highlights": highlights
        }

        file_exists = os.path.isfile(RESULT_FILE)
        with open(RESULT_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=new_record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_record)

        print(f"\nĐã lưu kết quả vào: {RESULT_FILE}")
    else:
        print("Không tìm thấy dữ liệu hoặc lỗi phân tích.")