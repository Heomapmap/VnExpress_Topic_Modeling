**VnExpress Topic Modeling and Trend Analysis**

Dự án thực hiện phân tích chủ đề và xu hướng trên dữ liệu bài báo từ VnExpress sử dụng kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và mô hình hóa chủ đề Latent Dirichlet Allocation (LDA).
Để phân loại các bài báo thành các chủ đề khác nhau, chúng ta sử dụng mô hình LDA, một kỹ thuật phổ biến trong NLP để phát hiện các chủ đề tiềm ẩn trong tập dữ liệu văn bản lớn.
Dự án này nhằm mục đích cung cấp cái nhìn sâu sắc về các xu hướng tin tức và chủ đề phổ biến trong các bài báo của VnExpress.
Đồng thời hỗ trợ phân tích xu hướng theo thời gian, giúp người dùng hiểu rõ hơn về sự thay đổi của các chủ đề tin tức qua các giai đoạn khác nhau.

**Project Structure**

```Project Structure
VnExpress_Topic_Modeling/
├── data/                     # Lưu trữ dữ liệu
│   ├── raw/                  # Dữ liệu thô vừa cào về
│   └── processed/            # Dữ liệu sạch sau khi tiền xử lý
├── models/                   # Lưu trữ mô hình LDA
│   └── lda/                  # Các file model (.model, .state, .dictionary)
├── reports/                  # Kết quả trực quan hóa
│   └── figures/              # Các biểu đồ (Hourly activity, Topic distribution, Wordclouds)
├── src/                      # Mã nguồn chính
│   ├── utils/                # Công cụ bổ trợ (scrape.py, summarize.py)
│   │   │── __init__.py       # Khởi tạo package utils
│   │   │── common_utils.py   # Hàm tiện ích chung cho cào dữ liệu
│   │   │── scrape_rss.py     # Cào dữ liệu từ RSS feed của VnExpress
│   │   │── scrape_web.py     # Cào dữ liệu trực tiếp từ trang web
│   │   │── summarize.py      # Tóm tắt bài báo bằng AI
│   │   └── vn_stopwords.txt  # Danh sách từ dừng tiếng Việt
│   ├── clean.py              # Tiền xử lý văn bản tiếng Việt
│   ├── lda_train.py          # Huấn luyện mô hình LDA
│   ├── predict_topic.py      # Dự đoán chủ đề cho bài báo mới
│   └── visualize.py          # Vẽ biểu đồ và phân tích xu hướng
├── README.md                 # Hướng dẫn dự án
└── requirements.txt          # Danh sách thư viện
```

**Hướng dẫn cài đặt và dọn dẹp môi trường:**
1. Tạo môi trường ảo .venv và cài đặt thư viện:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Windows sử dụng `venv\Scripts\activate`
   ```
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
3. Dọn dẹp môi trường (nếu cần):
   ```bash
   deactivate
   rm -rf venv
   ```
4. Nếu thiếu bộ phân giải HTML (lxml):
   ```bash
   pip install lxml
   ```
**Lưu ý:** Đảm bảo rằng bạn đã cài đặt tất cả các thư viện cần thiết trong `requirements.txt` để tránh lỗi khi chạy mã nguồn.

**Quy trình làm việc chính:**
1. **Cào dữ liệu:** Sử dụng `scrape_rss.py` hoặc `scrape_web.py` trong thư mục `src/utils/` để thu thập bài báo từ VnExpress và lưu vào `data/raw/`.
2. **Tiền xử lý dữ liệu:** Chạy `clean.py` để làm sạch và chuẩn hóa văn bản, lưu kết quả vào `data/processed/`.
3. **Huấn luyện mô hình LDA:** Sử dụng `lda_train.py` để huấn luyện mô hình LDA trên dữ liệu đã tiền xử lý và lưu mô hình vào thư mục `models/lda/`.
4. **Dự đoán chủ đề:** Sử dụng `predict_topic.py` để dự đoán chủ đề cho các bài báo mới.
5. **Phân tích và trực quan hóa:** Chạy `visualize.py` để tạo các biểu đồ phân tích xu hướng và lưu vào `reports/figures/`.

**Usage Examples:**
1.  **Cào dữ liệu từ RSS feed:**
  ```bash
  python src/utils/scrape_rss.py --output data/raw/rss_data.json
  ```
2.  **Làm sạch dữ liệu:**
  ```bash
  python src/clean.py --input data/raw/rss_data.json --output data/processed/cleaned_data.json
  ```
3.  **Huấn luyện mô hình LDA:**
  ```bash
  python src/lda_train.py --input data/processed/cleaned_data.json --model_output models/lda/lda_model.model
  ```
4.  **Dự đoán chủ đề cho bài báo mới:**
  ```bash
  python src/predict_topic.py --model models/lda/lda_model.model --text "Nội dung bài báo mới cần phân loại"
  ```
5.  **Trực quan hóa kết quả:**
  ```bash
  python src/visualize.py --input data/processed/cleaned_data.json --output reports/figures/
  ```

**Lưu ý:**
- Khi cào dữ liệu bằng `scrape_rss.py` dữ liệu thô lấy được sẽ sạch hơn nhưng giới hạn 60 bài/topic do hạn chế của RSS feed chỉ lưu các bài mới nhất.
- Khi cào dữ liệu bằng `scrape_web.py` dữ liệu thô lấy được sẽ nhiều hơn nhưng cần tiền xử lý kỹ hơn.
- Để chạy `predict_topic.py` bạn cần nhập api key groq của bạn vào `.env`.

**Kết quả trực quan hóa:**
- Biểu đồ hoạt động theo giờ (Hourly activity)
- Phân phối chủ đề (Topic distribution)
- Wordclouds cho từng chủ đề
