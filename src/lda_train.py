import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR.parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "clean_data.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "lda_train.csv"
MODEL_DIR = BASE_DIR / "models" / "lda"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def run_lda_pipeline():
    if not INPUT_FILE.exists():
        print(f"Lỗi: Không tìm thấy file {INPUT_FILE}. Vui lòng chạy clean.py trước.")
        return

    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=['clean_text', 'category'])

    num_topics = len(df['category'].unique())
    print(f"Hệ thống sẽ tìm kiếm {num_topics} chủ đề dựa trên dữ liệu...")

    texts = [str(text).split() for text in df["clean_text"]]
    dictionary = corpora.Dictionary(texts)

    dictionary.filter_extremes(no_below=5, no_above=0.3)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print(f"Đang huấn luyện LDA...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=20,
        iterations=200,
        alpha=0.01,
        eta=0.01,
        per_word_topics=True
    )

    lda_model.save(str(MODEL_DIR / "lda_model.model"))
    dictionary.save(str(MODEL_DIR / "id2word.dictionary"))

    def get_dominant_topic_info(bow):
        topics = lda_model.get_document_topics(bow)
        dominant = sorted(topics, key=lambda x: x[1], reverse=True)[0]
        return dominant[0], dominant[1]

    results = [get_dominant_topic_info(bow) for bow in corpus]
    df['topic_id'] = [res[0] for res in results]
    df['topic_confidence'] = [res[1] for res in results]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")


    print("\nĐang tính toán chỉ số Coherence (có thể mất một chút thời gian)...")
    coherence_model_lda = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    #Tính độ mạch lạc (coherence)
    coherence_score = coherence_model_lda.get_coherence()

    #Tính toán độ phức tạp (perplexity)
    perplexity_score = lda_model.log_perplexity(corpus)

    print("\n" + "=" * 50)
    print("                BÁO CÁO HUẤN LUYỆN LDA")
    print("=" * 50)
    print(f"[*] Số lượng Topics:      {lda_model.num_topics}")
    print(f"[*] Số lượng Từ vựng:     {len(dictionary)}")
    print(f"[*] Coherence Score (C_v): {coherence_score:.4f}  <-- (Mục tiêu: > 0.4 - 0.6)")
    print(f"[*] Perplexity:           {perplexity_score:.4f}  <-- (Càng thấp càng tốt)")
    print(f"[*] File Model đã lưu:    {MODEL_DIR / 'lda_model.model'}")
    print("=" * 50)

    print("\n[DANH SÁCH TỪ KHÓA CHI TIẾT THEO TOPIC]")
    for idx, topic in lda_model.print_topics(num_topics=-1, num_words=10):
        print(f"\nTOPIC ID {idx}:")
        words = topic.split("+")
        for w in words:
            print(f"   - {w.strip()}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    run_lda_pipeline()
