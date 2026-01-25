import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from wordcloud import WordCloud
from pathlib import Path
from gensim.models import LdaModel

from utils.common_utils import filter_by_time

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "processed" / "lda_train.csv"
MODEL_PATH = BASE_DIR / "models" / "lda" / "lda_model.model"
OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_topic_distribution(df_data):
    if df_data.empty: return
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_data, x='topic_id', hue='topic_id', palette='viridis', legend=False)
    plt.title('PHÂN BỔ SỐ LƯỢNG BÀI BÁO THEO CHỦ ĐỀ', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "topic_distribution.png")
    plt.show()


def generate_wordclouds(lda):
    num_topics = lda.num_topics
    cols = 4
    rows = (num_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))

    if num_topics > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    i = -1
    for i in range(num_topics):
        topic_words = dict(lda.show_topic(i, topn=30))
        cloud = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate_from_frequencies(
            topic_words)
        axes[i].imshow(cloud, interpolation='bilinear')

        title = f"Topic {i}"
        axes[i].set_title(title, fontsize=16, fontweight='bold')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wordclouds.png")
    plt.show()


def plot_hourly_activity(data_df):
    if data_df.empty:
        print("Không có dữ liệu để vẽ biểu đồ giờ")
        return

    df_plot = data_df.copy()

    df_plot = df_plot.dropna(subset=['publish_hour'])
    df_plot['hour_only'] = df_plot['publish_hour'].str.split(':').str[0].astype(int)

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df_plot, x='hour_only', color='skyblue')

    plt.title('THỜI ĐIỂM ĐĂNG BÀI TRONG NGÀY', fontsize=14, fontweight='bold')
    plt.xlabel('Giờ trong ngày (0h - 23h)')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hourly_activity.png")
    plt.show()


if __name__ == "__main__":
    if not INPUT_CSV.exists() or not MODEL_PATH.exists():
        print("Lỗi: Không tìm thấy file dữ liệu hoặc model.")
    else:
        df = pd.read_csv(INPUT_CSV)

        df_filtered = filter_by_time(df)

        if df_filtered.empty:
            print("Không tìm thấy dữ liệu cho khoảng thời gian này.")
        else:
            lda_model = LdaModel.load(str(MODEL_PATH))
            print(f"Đang xử lý {len(df_filtered)} bài báo...")

            print("\nVẽ biểu đồ phân bố chủ đề...")
            time.sleep(2)
            plot_topic_distribution(df_filtered)

            print("Tạo đám mây từ ngữ cho các chủ đề...")
            time.sleep(2)
            generate_wordclouds(lda_model)

            print("Vẽ biểu đồ hoạt động theo giờ đăng bài...")
            time.sleep(2)
            plot_hourly_activity(df_filtered)