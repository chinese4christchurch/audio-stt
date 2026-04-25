# Audio STT 處理工具 這是一套用於處理講道音檔下載、AI 摘要生成與資料匯出的工具腳本。

## 專案結構
- `1_download.py`: 從 S3 下載資料庫中標記為 Active 但本地缺失的音檔。
- `2_summarize.py`: 讀取字幕檔並呼叫 Gemini 1.5 Pro 產生繁簡中文摘要。
- `3_export.py`: 將更新後的資料匯出為 SQL 指令。

## 前置準備

### 1. 安裝 Python 必要套件
本專案需要以下 Python 函式庫：
```bash
pip3 install mysql-connector-python google-generativeai python-dotenv requests --break-system-packages
```

### 2. 環境變數設定
請確保專案根目錄（`audio-stt` 的上一層）存在 `.env` 檔案，並包含以下內容：
```env
# 資料庫資訊 (程式中已預設連線 127.0.0.1:3307)
MYSQL_DATABASE=wordpress_資料庫名稱
MYSQL_USER=wordpress_資料庫用戶名
MYSQL_PASSWORD=password_資料庫密碼

# Gemini API Key (必要)
GOOGLE_API_KEY=您的_Gemini_API_Key
```

## 使用方法

所有的程式請在 **專案根目錄** 執行：

### 步驟 1：下載音檔
下載遺失的 `.mp3` 檔案到 `sermon/mp3/` 目錄。
```bash
python3 audio-stt/1_download.py
```

### 步驟 2：產生 AI 摘要
讀取 `sermon/mp3/` 下的 `.srt` 或 `.txt` 檔案，呼叫 Gemini 3.5 latest 產生摘要並更新至資料庫。
- 每筆處理前會暫停 10 秒供使用者確認/中斷。
- 每筆處理後會強制等待 30 秒以符合 API 免費額度限制。
```bash
python3 audio-stt/2_summarize.py
```

### 步驟 3：匯出 SQL
將資料庫中已完成摘要（content 長度 > 99）的資料匯出至 `export.sql`。
```bash
python3 audio-stt/3_export.py
```

## 注意事項
- **節流保護**：`2_summarize.py` 採用單線程且設有嚴格的等待時間，請勿隨意縮短等待時間以免觸發 API 封鎖。
- **斷點續傳**：摘要程式會自動跳過 `note` 欄位已有資料的項目，若中斷可直接重啟。
- **檔案支援**：摘要程式優先尋找同名的 `.srt` 檔，若無則尋找 `.txt` 檔。
