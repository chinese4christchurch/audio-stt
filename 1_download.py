import os
import mysql.connector
import requests
from pathlib import Path
import time

# 資料庫連線資訊
# 卻認當前目錄下有.env檔案如下
# MYSQL_DATABASE=你的wordpress資料庫名稱
# MYSQL_USER=你的wordpress資料庫用戶名
# MYSQL_PASSWORD=你的password資料庫密碼
# MYSQL_PORT=你的wordpress資料庫port 如3306
# MYSQL_HOST=你的wordpress資料庫ip 如127.0.0.1
# GOOGLE_API_KEY=你的Google API key
DB_CONFIG = {
    'host': os.getenv("MYSQL_HOST"),
    'port': int(os.getenv("MYSQL_PORT")),
    'user': os.getenv("MYSQL_USER"),
    'password': os.getenv("MYSQL_PASSWORD"),
    'database': os.getenv("MYSQL_DATABASE")
}

BASE_URL = "https://s3-us-west-1.amazonaws.com/chinese-church/restructure_sermon"
BASE_DIR = Path(__file__).resolve().parent.parent
SERMON_DIR = BASE_DIR / 'sermon' / 'mp3'

def download_file(url, dest):
    """下載單一檔案，失敗時捕捉異常並回傳 False，不中斷程式"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"\n   [警告] 下載失敗 ({url}): {e}")
        return False

def main():
    print("=== [1. 下載程式] 啟動 ===")
    if not SERMON_DIR.exists():
        SERMON_DIR.mkdir(parents=True, exist_ok=True)

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # 使用您指定的 Query: 找出 Active 且 content 為空的資料來下載音檔
        query = """
            SELECT DISTINCT(audiofile) 
            FROM wp_audio_list 
            WHERE activeFlag = 'Active' 
              AND audiofile IS NOT NULL 
              AND audiofile != '' 
              AND content IS NULL
        """
        print("正在執行查詢...")
        cursor.execute(query)
        rows = cursor.fetchall()
        
        total = len(rows)
        print(f"找到 {total} 筆符合條件的資料。")

        download_count = 0
        fail_count = 0

        for idx, row in enumerate(rows, 1):
            filename = row['audiofile']
            if not filename or len(filename) < 4: continue
            
            year_str = filename[:4]
            if year_part := year_str.isdigit():
                year = int(year_str)
                if 1978 <= year <= 2026:
                    dest_path = SERMON_DIR / filename
                    
                    if not dest_path.exists():
                        print(f"({idx}/{total}) 準備下載: {filename}")
                        url = f"{BASE_URL}/{year}/{filename}"
                        if download_file(url, dest_path):
                            print(f"   [成功] 已存至 {dest_path}")
                            download_count += 1
                        else:
                            fail_count += 1
                        
                        # 每下載一筆休息 1 秒，保持禮貌
                        time.sleep(1)
                    else:
                        # 檔案已存在則跳過
                        if idx % 100 == 0:
                            print(f"進度: 已檢查 {idx}/{total} 筆 (檔案已存在)...")

        cursor.close()
        conn.close()
        print("\n=== 下載作業結束 ===")
        print(f"總計檢查: {total} 筆")
        print(f"本次下載: {download_count} 筆")
        print(f"下載失敗: {fail_count} 筆")

    except Exception as e:
        print(f"發生嚴重錯誤: {e}")

if __name__ == "__main__":
    main()
