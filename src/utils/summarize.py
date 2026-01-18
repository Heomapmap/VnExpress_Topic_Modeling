import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_ai_insight(title, description, content):
    if not content or len(content) < 50:
        return "Nội dung quá ngắn.", ""

    prompt = f"""
    Hãy phân tích bài báo sau bằng tiếng Việt:
    Tiêu đề: {title}
    Nội dung: {content[:3000]} 

    Yêu cầu trả về chính xác theo định dạng sau (không viết thêm lời dẫn):
    [TÓM TẮT]
    (Nội dung tóm tắt khoảng 2-3 câu)
    [Ý CHÍNH]
    (Danh sách các gạch đầu dòng ý chính)
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Hoặc llama3-8b-8192
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # Giảm độ sáng tạo để AI tuân thủ định dạng tốt hơn
        )

        response_text = completion.choices[0].message.content

        summary = ""
        highlights = ""

        if "[TÓM TẮT]" in response_text and "[Ý CHÍNH]" in response_text:
            summary = response_text.split("[TÓM TẮT]")[1].split("[Ý CHÍNH]")[0].strip()
            highlights = response_text.split("[Ý CHÍNH]")[1].strip()
        else:
            summary = response_text.strip()
            highlights = ""

        return summary, highlights
    except Exception as e:
        return f"Lỗi Groq: {e}", ""