import mysql.connector
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent.parent
EXPORT_FILE = BASE_DIR / 'export.sql'

def escape_sql_string(s):
    if s is None:
        return "NULL"
    # 基本的 SQL 逸脫處理
    return "'" + s.replace("'", "''").replace("\\", "\\\\") + "'"

def main():
    print("=== [3. 匯出程式] 啟動 ===")
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # 選取內容大於 99 字的
        query = "SELECT audiofile, note, content FROM wp_audio_list WHERE activeFlag = 'Active' AND LENGTH(content) > 99"
        cursor.execute(query)
        rows = cursor.fetchall()
        
        print(f"找到 {len(rows)} 筆符合匯出條件的資料。")
        
        with open(EXPORT_FILE, 'a', encoding='utf-8') as f:
            for row in rows:
                note_sql = escape_sql_string(row['note'])
                content_sql = escape_sql_string(row['content'])
                audiofile_sql = escape_sql_string(row['audiofile'])
                print(f"audiofile {audiofile_sql} 匯出中...")
                sql_line = f"UPDATE wp_audio_list SET note={note_sql}, content={content_sql} WHERE activeFlag = 'Active' AND audiofile={audiofile_sql};\n"
                f.write(sql_line)
        
        print(f"匯出完成！檔案位於: {EXPORT_FILE}")
        
        cursor.close()
        conn.close()

    except Exception as e:
        print(f"匯出失敗: {e}")

if __name__ == "__main__":
    main()
