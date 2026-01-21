import pandas as pd
import numpy as np
import gensim
import ast
from pathlib import Path

from utils.common_utils import cosine_similarity, get_topic_vector

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "lda" / "lda_model.model"
DICT_PATH = BASE_DIR / "models" / "lda" / "id2word.dictionary"
DATA_PATH = BASE_DIR / "data" / "processed" / "lda_train.csv"

class ArticleRecommender:
    def __init__(self):
        print("Đang tải dữ liệu & mô hình...")
        self.df = pd.read_csv(DATA_PATH)
        self.lda_model = gensim.models.LdaModel.load(str(MODEL_PATH))
        self.dictionary = gensim.corpora.Dictionary.load(str(DICT_PATH))
        self.df['topic_vector'] = self.df['topic_vector'].apply(ast.literal_eval)
        self.matrix = np.vstack(self.df['topic_vector'].values)

    def recommend_articles(self, input_text, top_n=5):

        #Chuyển input người dùng thành vector
        input_vec = get_topic_vector(input_text, self.lda_model, self.dictionary)

        #Tính độ tương đồng giữa input và toàn bộ kho baì báo
        sims = cosine_similarity(input_vec, self.matrix)

        #Lấy top n kết quả cao nhất
        top_idx = np.argsort(sims)[::-1][:top_n]

        return self.df.iloc[top_idx][["title", "category", "url", "clean_text"]], sims[top_idx]

if __name__ == "__main__":
    recommender = ArticleRecommender()

    while True:
        print("\n" + "=" * 50)
        text_input = input("Nhập nội dung/tiêu đề để tìm bài viết tương tự (hoặc 'q' để thoát): ").strip()
        if text_input.lower() == 'q': break

        recs, scores = recommender.recommend_articles(text_input, top_n=5)

        print(f"\nTop 5 bài viết tương tự nhất với: '{text_input[:50]}...'")
        for i, (row, score) in enumerate(zip(recs.itertuples(), scores), start=1):
            print(f"{i}. [{row.category.upper()}] {row.title}")
            print(f"   Độ tương đồng: {score:.2%} | Link: {row.url}")