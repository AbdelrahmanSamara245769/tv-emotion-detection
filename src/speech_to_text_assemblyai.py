"""
speech_to_text_assemblyai.py

Transcribes an MP3 audio file with AssemblyAI, splits the text into sentences,
and saves one sentence per row in an Excel file.

Needs the ASSEMBLYAI_API_KEY environment variable (see .env.example).

Usage:
    python src/speech_to_text_assemblyai.py episode.mp3 --out transcribed_data_assemblyAI.xlsx
"""

import argparse
import os
import re

import assemblyai as aai
import pandas as pd


def main(mp3_file, output_file="transcribed_data_assemblyAI.xlsx"):
    """
    Transcribe an mp3 audio file using AssemblyAI and save as xlsx.
    """
    aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

    #  Create the transcriber object
    transcriber = aai.Transcriber()

    #  Transcribe the audio file (no language_code needed)
    transcript = transcriber.transcribe(mp3_file)

    # Extract text
    text = transcript.text

    # Split text into sentences
    text = re.sub(r'[!?]', '.', text)
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    # Save to xlsx
    df = pd.DataFrame(sentences, columns=["Sentence"])
    df.to_excel(output_file, index=False)

    print(f"Transcription saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an MP3 file with AssemblyAI.")
    parser.add_argument("mp3_file")
    parser.add_argument("--out", default="transcribed_data_assemblyAI.xlsx")
    args = parser.parse_args()
    main(args.mp3_file, args.out)
