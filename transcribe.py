"""
Transcribe audio files to SRT subtitle using OpenAI Whisper.

Requirements:
     on the system such as `brew install ffmpeg` on Mac, `apt install ffmpeg` on Linux or https://ffmpeg.org/download.html on windows
    1. pip install whisper opencc-python-reimplemented
    2. ffmpeg
       Mac:     `brew install ffmpeg` or https://evermeet.cx/ffmpeg/
       Linux:   `apt-get install ffmpeg`
       Windows: https://ffmpeg.org/download.html

Suggestion:
    numpy should be latest 1.x, such as `pip install "numpy<2"`

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
"""

import time
import os
import whisper
import argparse
import opencc

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def transcribe_to_srt(audio_path, to_simplified=False, model_name="large"):
    if not os.path.exists(audio_path):
        print(f"Audio file '{audio_path}' does not exist.")
        return

    base_name = os.path.splitext(audio_path)[0]
    output_file = f"{base_name}.srt"
    if os.path.exists(output_file):
        print(f"Subtitle file '{output_file}' already exist.")
        return

    print(f"Loading {model_name} model on cpu")
    model = whisper.load_model(model_name)
    print(f"Starting transcribing {audio_path}")
    start_time = time.time()
    result = model.transcribe(audio_path, language="zh")

    if to_simplified:
        try:
            from opencc import OpenCC
            cc = OpenCC('t2s')
        except ImportError:
            print("OpenCC not installed. Run `pip install opencc-python-reimplemented` to use --to-simplified.")
            cc = None
    else:
        try:
            from opencc import OpenCC
            cc = OpenCC('s2t')
        except ImportError:
            print("OpenCC not installed. Run `pip install opencc-python-reimplemented` to use --to-simplified.")
            cc = None

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], start=1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            if cc:
                text = cc.convert(text)
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    elapsed_time = time.time() - start_time
    print(f"Finished transcribing to {output_file}, took {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio to SRT using Whisper")
    parser.add_argument("audio_path", help="Path to the audio file (e.g., .mp3 or .wav)")
    parser.add_argument("--to-simplified", action="store_true", help="Convert Traditional Chinese to Simplified Chinese")
    parser.add_argument("--model", default="large", choices=["tiny", "base", "small", "medium", "large"],
                    help="Whisper model size to use (default: large)")

    args = parser.parse_args()

    transcribe_to_srt(args.audio_path, to_simplified=args.to_simplified, model_name=args.model)
