import pandas as pd
from pathlib import Path
from tqdm import tqdm

from utils.common_utils import clean_text

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR.parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "data_web.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "clean_data.csv"

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"Lỗi: Không tìm thấy file tại {INPUT_FILE}")
        exit()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nKích thước ban đầu: {df.shape}")

    df = df.dropna(subset=["category", "title", "content"])
    df.drop_duplicates(subset=["title", "content"], inplace=True)
    df = df[df["content"].str.strip().str.len() > 20]

    df["full_text"] = df["title"].fillna("") + " " + df["description"].fillna("") + " " + df["content"]

    tqdm.pandas(desc="Đang làm sạch dữ liệu")

    df["clean_text"] = df["full_text"].progress_apply(lambda x: clean_text(x, return_list=False))
    df = df[df["clean_text"].str.len() > 10]

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nHoàn thành! Số lượng sau khi lọc: {df.shape}")
    print(f"File lưu tại: {OUTPUT_FILE}")