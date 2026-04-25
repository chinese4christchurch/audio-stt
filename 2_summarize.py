import os
import time
import mysql.connector
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

# 資料庫連線資訊
# 卻認當前目錄下有.env檔案如下
# MYSQL_DATABASE=你的wordpress資料庫名稱
# MYSQL_USER=你的wordpress資料庫用戶名
# MYSQL_PASSWORD=你的password資料庫密碼
# MYSQL_PORT=你的wordpress資料庫port 如3306
# MYSQL_HOST=你的wordpress資料庫ip 如127.0.0.1
# GOOGLE_API_KEY=你的Google API key

# --- 第一階段：環境檢查 ---
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("【錯誤】找不到 GOOGLE_API_KEY。")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# 2026 年建議嘗試的免費額度模型順序
TRY_MODELS = [
    'models/gemini-3.1-flash-lite-preview',
    'models/gemini-flash-lite-latest',
    'models/gemini-2.5-flash-lite',
    'models/gemini-pro-latest'
]

def get_working_model():
    print("正在偵測可用的免費模型...")
    for model_name in TRY_MODELS:
        try:
            print(f"嘗試連結 {model_name}...", end=' ', flush=True)
            m = genai.GenerativeModel(model_name)
            # 測試極短的請求以確認額度
            m.generate_content("hi", generation_config={"max_output_tokens": 1})
            print("成功！")
            return m
        except Exception as e:
            print("失敗 (無額度或不支援)")
    return None

model = get_working_model()
if not model:
    print("\n【嚴重錯誤】所有 Flash/Lite 模型均回傳 limit: 0 或 429。")
    print("這通常代表您的 Google Cloud 專案尚未開啟 Generative Language API 的免費層級。")
    print("請前往 https://aistudio.google.com/ 檢查計畫狀態。")
    exit(1)

DB_CONFIG = {
    'host': os.getenv("MYSQL_HOST"),
    'port': int(os.getenv("MYSQL_PORT")),
    'user': os.getenv("MYSQL_USER"),
    'password': os.getenv("MYSQL_PASSWORD"),
    'database': os.getenv("MYSQL_DATABASE")
}

SERMON_DIR = BASE_DIR / 'sermon' / 'mp3'

def get_summary_from_ai(speaker, topic, section, content):
    prompt = f"""
這是講員{speaker}講道的逐字稿, 主題是{topic}, """
    if section:
        prompt += f"基於基督教聖經 {section}, "
    
    prompt += f"""幫我產生摘要。
逐字稿內容：
{content}

請嚴格依照以下格式輸出兩份摘要，每份摘要約 1500 字左右：
---TRADITIONAL---
AI摘要有錯歡迎指正:

[這裡放入繁體中文摘要]

---SIMPLIFIED---
AI摘要有错欢迎指正:

[這裡放入簡體中文摘要]
"""

    try:
        response = model.generate_content(prompt)
        text = response.text
        
        trad = ""
        simp = ""
        if "---TRADITIONAL---" in text and "---SIMPLIFIED---" in text:
            parts = text.split("---SIMPLIFIED---")
            trad = parts[0].replace("---TRADITIONAL---", "").strip()
            simp = parts[1].strip()
        return trad, simp
    except Exception as e:
        print(f"AI 呼叫失敗: {e}")
        return None, None

def main():
    print(f"=== [2. 摘要程式] 啟動 ===")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT audiofile, speaker, topic, section 
            FROM wp_audio_list 
            WHERE activeFlag = 'Active' 
              AND content IS NULL 
              AND audiofile != '' 
              AND audiofile IS NOT NULL
              AND (note IS NULL OR note = '')
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        total = len(rows)
        print(f"找到 {total} 筆待處理資料。")

        for idx, row in enumerate(rows, 1):
            audiofile = row['audiofile']
            speaker = row['speaker'] or "未知講員"
            topic = row['topic'] or "未知主題"
            section = row['section'] or ""
            
            base_name = Path(audiofile).stem
            srt_path = SERMON_DIR / f"{base_name}.srt"
            if not srt_path.exists():
                srt_path = SERMON_DIR / f"{base_name}.txt"

            if not srt_path.exists():
                continue

            print("-" * 40)
            print(f"({idx}/{total}) 處理中: {audiofile} {speaker} {section} {topic}")
            print(f"{time.ctime()} 等待 10 秒供中斷...")
            time.sleep(10)

            with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
                srt_content = f.read()

            print("呼叫 AI 產生摘要...")
            trad_summary, simp_summary = get_summary_from_ai(speaker, topic, section, srt_content)

            if trad_summary and simp_summary:
                update_query = "UPDATE wp_audio_list SET note = %s, content = %s WHERE audiofile = %s"
                cursor.execute(update_query, (trad_summary, simp_summary, audiofile))
                conn.commit()
                print(f"[成功] 已更新資料庫。")
                print(f"節流等待 30 秒...")
                time.sleep(30)
            else:
                print(f"[失敗] 無法產生摘要。")

        cursor.close()
        conn.close()

    except KeyboardInterrupt:
        print("\n中斷。")
    except Exception as e:
        print(f"錯誤: {e}")

if __name__ == "__main__":
    main()
