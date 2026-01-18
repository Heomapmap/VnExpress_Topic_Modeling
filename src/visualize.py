import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path
from gensim.models import LdaModel

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "processed" / "lda_train.csv"
MODEL_PATH = BASE_DIR / "models" / "lda" / "lda_model.model"
OUTPUT_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def filter_by_time(df):
    print("\n--- BỘ LỌC THỜI GIAN ---")
    print("(Nhấn Enter để bỏ qua bộ lọc và xem toàn bộ dữ liệu)")

    year_input = input("Nhập năm muốn xem: ").strip()
    month_input = input("Nhập tháng muốn xem (1-12): ").strip()

    df = df.copy()
    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    df = df.dropna(subset=['publish_date'])

    filtered_df = df

    if year_input.isdigit():
        filtered_df = filtered_df[filtered_df['publish_date'].dt.year == int(year_input)]
        print(f"Đã lọc theo năm: {year_input}")
    else:
        print("Không lọc theo năm (Xem tất cả các năm)")

    if month_input.isdigit():
        filtered_df = filtered_df[filtered_df['publish_date'].dt.month == int(month_input)]
        print(f"Đã lọc theo tháng: {month_input}")
    else:
        print("Không lọc theo tháng (Xem tất cả các tháng)")

    return filtered_df


def plot_topic_distribution(df):
    if df.empty: return
    plt.figure(figsize=(10, 6))
    df['topic_name'] = df['topic_id']
    sns.countplot(data=df, x='topic_name', hue='topic_name', palette='viridis', legend=False)
    plt.title('PHÂN BỔ SỐ LƯỢNG BÀI BÁO THEO CHỦ ĐỀ', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "topic_distribution.png")
    plt.show()


def generate_wordclouds(lda_model, topic_mapping=None):
    """Vẽ đám mây từ ngữ"""
    num_topics = lda_model.num_topics
    cols = 2
    rows = (num_topics + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()

    for i in range(num_topics):
        topic_words = dict(lda_model.show_topic(i, topn=30))
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


def plot_hourly_activity(df):
    if df.empty: return
    df_plot = df.copy()

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

        # 1. Lọc thời gian trước
        df_filtered = filter_by_time(df)

        if df_filtered.empty:
            print("Không tìm thấy dữ liệu cho khoảng thời gian này.")
        else:
            lda_model = LdaModel.load(str(MODEL_PATH))
            print(f"Đang xử lý {len(df_filtered)} bài báo...")

            print("\nVẽ biểu đồ phân bố chủ đề...")
            time_sleep=2
            plot_topic_distribution(df_filtered)

            print("Tạo đám mây từ ngữ cho các chủ đề...")
            time_sleep=2
            generate_wordclouds(lda_model)

            print("Vẽ biểu đồ hoạt động theo giờ đăng bài...")
            time_sleep=2
            plot_hourly_activity(df_filtered)