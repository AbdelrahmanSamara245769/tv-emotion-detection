"""
speech_to_text_whisper.py

Transcribes an MP3 audio file with Whisper (small) and saves one sentence per row
in a CSV file. Needs a CUDA GPU.

Usage:
    python src/speech_to_text_whisper.py episode.mp3 --out transcribed_data_whisper.csv
"""

import argparse

import nltk
import pandas as pd
import whisper

# Make sure punkt is available once
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def main(audio_file: str, outputfile: str = "transcribed_data_whisper.csv"):
    """
    Transcribe a MP3 file into a csv using Whisper where each sentence is separated
    """
    # load whisper model
    model = whisper.load_model("small").to("cuda")

    # Transcribe
    result = model.transcribe(audio_file)
    transcript = result["text"]

    # Split the transcription into sentences
    sentences = nltk.sent_tokenize(transcript)

    # save to a csv file
    df = pd.DataFrame(sentences, columns=["Sentence"])
    df.to_csv(outputfile, index=False, encoding="utf-8")

    print(f"Saved {len(sentences)} sentences → {outputfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an MP3 file with Whisper (small).")
    parser.add_argument("audio_file")
    parser.add_argument("--out", default="transcribed_data_whisper.csv")
    args = parser.parse_args()
    main(args.audio_file, args.out)
