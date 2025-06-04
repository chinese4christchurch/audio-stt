"""
Transcribe audio files to SRT subtitle using OpenAI Whisper.

Requirements:
     on the system such as `brew install ffmpeg` on Mac, `apt install ffmpeg` on Linux or https://ffmpeg.org/download.html on windows
    1. pip install whisper opencc-python-reimplemented pypinyin rapidfuzz
    2. in some cases the default whisper installed wrong version and caused error of module 'whisper' has no attribute 'load_model'
       pip uninstall whisper
       pip install git+https://github.com/openai/whisper.git
    3. ffmpeg with ffprobe must be installed system-wide
       Mac:     `brew install ffmpeg` or https://evermeet.cx/ffmpeg/
       Linux:   `apt-get install ffmpeg`
       Windows: https://ffmpeg.org/download.html

Suggestion:
    If there's some numpy error on the old python, please downgrade numpy to the latest 1.x, such as `pip install "numpy<2"`

Usage:
    python transcribe.py <audio_path> [--to-simplified] [--model MODEL_NAME]

Arguments:
    audio_path           Path to the audio file (e.g., .mp3 or .wav)

Optional Flags:
    --to-simplified      Convert Traditional Chinese to Simplified Chinese
    --model              Whisper model to use: tiny, base, small, medium, large (default: large)

Description:
    This script loads a specified Whisper model and transcribes the given audio
    file into an SRT subtitle file. It optionally converts the text between
    Traditional and Simplified Chinese using OpenCC.
    Also corrects commonly mis-transcribed homophones based on known Chinese phrases.
"""
import os
import time
import argparse
import subprocess
import whisper
import opencc
import tempfile
import shutil
from rapidfuzz import fuzz
from pypinyin import lazy_pinyin

def pinyin_equal(a, b):  #compare terms strictly
    return lazy_pinyin(a) == lazy_pinyin(b)

def get_audio_duration(path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Failed to get duration with ffprobe: {e}")
        return None

def is_video_file(filepath):
    video_exts = ['.mp4', '.mkv', '.mov', '.avi']
    return os.path.splitext(filepath)[1].lower() in video_exts

def extract_audio_from_video(video_path):
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "extracted.wav")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return audio_path, temp_dir
    except Exception as e:
        print(f"Failed to extract audio: {e}")
        shutil.rmtree(temp_dir)
        return None, None

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

known_phrases = [   # For correcting terms
    "創世記", "出埃及記", "利未記", "民數記", "申命記",
    "約書亞記", "士師記", "路得記", "撒母耳記上", "撒母耳記下",
    "列王紀上", "列王紀下", "歷代志上", "歷代志下", "以斯拉記",
    "尼希米記", "以斯帖記", "約伯記", "詩篇", "箴言",
    "傳道書", "雅歌", "以賽亞書", "耶利米書", "耶利米哀歌",
    "以西結書", "但以理書", "何西阿書", "約珥書", "阿摩司書",
    "俄巴底亞書", "約拿書", "彌迦書", "那鴻書", "哈巴谷書",
    "西番雅書", "哈該書", "撒迦利亞書", "瑪拉基書", "馬太福音",
    "馬可福音", "路加福音", "約翰福音", "使徒行傳", "羅馬書",
    "哥林多前書", "哥林多後書", "加拉太書", "以弗所書", "腓立比書",
    "歌羅西書", "帖撒羅尼迦前書", "帖撒羅尼迦後書", "提摩太前書", "提摩太後書",
    "提多書", "腓利門書", "希伯來書", "雅各書", "彼得前書",
    "彼得後書", "約翰一書", "約翰二書", "約翰三書", "猶大書",
    "啟示錄",
    "經文", "經節", "海沃", "詩班", "領詩", "臨到", "查經",
    "外邦人", "事奉", "靈修", "諮商",
]

def replace_with_known_phrases(text):
    words = text.split()
    for phrase in known_phrases:
        if any(pinyin_equal(phrase, w) or phrase in w for w in words):
            return phrase
    return text

def transcribe_to_srt(input_path, to_simplified=False, model_name="large", language="zh"):
    if not os.path.exists(input_path):
        print(f"File '{input_path}' does not exist.")
        return

    base_name = os.path.splitext(input_path)[0]
    output_file = f"{base_name}.srt"
    if os.path.exists(output_file):
        print(f"Subtitle file '{output_file}' already exists.")
        return

    if is_video_file(input_path):
        audio_path, temp_dir = extract_audio_from_video(input_path)
        if not audio_path:
            print("No audio files converted.")
            return
        print(f"A temporary audio file {audio_path} converted.")
    else:
        audio_path = input_path
        temp_dir = None

    print(f"Loading model: {model_name}")
    model = whisper.load_model(model_name)
    print(f"Transcribing: {input_path}")
    start_time = time.time()

    extra_args = {"language": language}
    if model_name == "large":
        extra_args["task"] = "transcribe"

    result = model.transcribe(audio_path, **extra_args)
    print(f"Transcribing finshed in {time.time() - start_time}s. Let's process.")
    duration = get_audio_duration(audio_path)

    cc = None
    try:
        cc = opencc.OpenCC('t2s' if to_simplified else 's2t')
    except:
        print("OpenCC not installed.")

    previous_text = None
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], start=1):
            text = segment['text'].strip()
            # print(f"processing raw text {i}: {text}")
            if cc:
                text = cc.convert(text)

            if text == previous_text:
                # print(f"skipping repeating text: {previous_text}")
                continue
            previous_text = text

            for phrase in known_phrases:
                if pinyin_equal(phrase, text):
                    text = phrase
                    break

            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    elapsed = time.time() - start_time
    if duration:
        speedup = duration / elapsed
        print(f"Completed in {elapsed:.2f}s for {duration:.2f}s audio ({speedup:.2f}x real-time)")
    else:
        print(f"Completed in {elapsed:.2f}s")

    if temp_dir:
        shutil.rmtree(temp_dir)
        print(f"The temporary audio file {audio_path} removed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio/video to SRT using Whisper")
    parser.add_argument("input_path", help="Path to audio/video file (e.g., .mp3, .mp4)")
    parser.add_argument("--to-simplified", action="store_true", help="Convert Traditional Chinese to Simplified")
    parser.add_argument("--model", default="large", choices=["tiny", "base", "small", "medium", "large"], help="Model size")
    parser.add_argument("--language", default="zh", help="Language (default: zh)")
    args = parser.parse_args()

    transcribe_to_srt(args.input_path, to_simplified=args.to_simplified, model_name=args.model, language=args.language)
