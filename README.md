# Emotion Detection Pipeline for TV Dialogue

This pipeline detects emotions in reality TV dialogue. It transcribes an episode, translates the sentences, and labels each one with one of seven emotions: happiness, sadness, anger, surprise, fear, disgust or neutral.

It was a group project for a team of three at Breda University of Applied Sciences (Applied Data Science & AI, Year 2), in September and October 2025. The client was the Content Intelligence Agency.

## The problem

Media analysts want to know the emotional tone of TV content, sentence by sentence, without labelling everything by hand. That requires reliable transcripts of noisy, conversational speech. It also requires a classifier that copes with short, ambiguous sentences and a strong imbalance between emotions: most lines are neutral or happy, and disgust is rare.

## How it works

```
episode audio
  -> AssemblyAI transcript, split into sentences
  -> English -> Dutch -> English translation (custom BART-style models)
  -> RoBERTa-Large emotion classifier (class-weighted loss)
  -> one emotion label per sentence (CSV)
```

1. **Speech to text.**
   - Two scripts transcribe the episode with AssemblyAI and with Whisper (small).
   - Word error rate was counted by hand: substitutions, insertions and deletions over 180 lines.
2. **NLP features:**
   - POS tags (spaCy)
   - TF-IDF
   - VADER sentiment
   - pretrained word embeddings
   - a custom Word2Vec model trained on the training corpus
   - emotion-lexicon counts
3. **Data quality report.** The labelled test transcript (665 sentences) is profiled for:
   - emotion balance (31% neutral, under 1% disgust)
   - sentence length and vocabulary
   - fillers, slang and transcription artefacts

   It ends with recommendations for choosing training data.
4. **Model iterations:**
   - **Traditional:** logistic regression, Naive Bayes and SVM on TF-IDF, sentiment and POS features.
   - **Recurrent:** RNN and LSTM models.
   - **Transformers:** BERT, RoBERTa (base and large, with VADER features and class weights) and DeBERTa-v3-large.
   - **Training data:** two candidate datasets were compared.
5. **Machine translation.** English ↔ Dutch sequence-to-sequence models trained from scratch on OpenSubtitles:
   - **Cleaning:** length and ratio rules, language identification, multilingual embedding similarity of at least 0.60, and deduplication, leaving about 1.2M training pairs.
   - **Tokenizer:** a 32k byte-level BPE vocabulary.
   - **Model:** a BART-style architecture with 8 encoder and 8 decoder layers, about 120M parameters, trained in bf16.
   - **Post-processing:** a domain glossary.
6. **Prompt engineering.** Llama 3.3 70B classifies sentences through an OpenAI-compatible API. Approaches, from simplest to most complex:
   - a zero-shot baseline
   - a structured two-pass prompt that escalates neutral answers
   - TF-IDF retrieval of few-shot examples, JSON probability outputs, class priors and temperature calibration
   - a hybrid of the two, with a happiness/surprise resolver

   All of them are evaluated on separate dev and test splits.
7. **Error analysis, explainability and model card.** Confusion analysis of the final model, explained with three methods: Gradient × Input, layer-wise relevance propagation and input perturbation. The model card is in [docs/model_card.md](docs/model_card.md).
8. **Final pipeline.** A notebook chains transcription, translation and classification.

## Results

| Component | Result |
|---|---|
| Speech to text (WER, 180 lines) | **AssemblyAI 3.84%**, Whisper small 5.02% |
| Final classifier: RoBERTa-Large, class-weighted (client test set) | **0.693 accuracy, 0.632 macro F1**, 0.692 weighted F1 |
| Best traditional model: SVM, TF-IDF bigrams + sentiment + POS | 0.29 test macro F1 (0.42 when trained on the smaller, in-domain dataset) |
| English → Dutch translation (full validation set) | **BLEU 34.7, chrF2 52.1** (29.4 / 47.2 with fast greedy decoding on 2,000 sentences) |
| Prompt engineering, Llama 3.3 70B (test split) | **0.628 macro F1** for the hybrid prompt, against 0.493 zero-shot |

**Transformer comparison** (test set)

| Model | Accuracy | Macro F1 |
|---|---|---|
| BERT-base | 0.588 | 0.483 |
| RoBERTa-base | 0.616 | 0.561 |
| RoBERTa-base + VADER | 0.613 | 0.588 |
| RoBERTa-Large | 0.672 | 0.596 |
| **RoBERTa-Large, class-weighted** | **0.693** | **0.632** |
| DeBERTa-v3-large, weighted + augmented | 0.652 | 0.554 |

**What we learned**

- **In-domain data:** a smaller dataset similar to the client's dialogue beat a larger, differently styled one for every traditional model, and it was used for the final transformers.
- **Class weighting:** it gave the biggest gain on rare emotions.
- **Prompt scores:** dev scores of up to 0.84 macro F1 were inflated by cached LLM responses. The test split gives the realistic picture: 0.49–0.63.

## My role

In a team of three, my parts were:

- **Speech to text:** the comparison of AssemblyAI and Whisper, including the hand-counted word error rates.
- **NLP features:** the feature extraction notebook.
- **Data quality report:** the profile of the client's test transcript.
- **Machine translation:** the English ↔ Dutch models, covering data cleaning, the tokenizer, training and evaluation.
- **Prompt engineering:** all three rounds, from the zero-shot baseline to the calibrated hybrid prompts.
- **Final pipeline:** the notebook that chains transcription, translation and classification into one run.

My teammates mainly built the model iterations (traditional, RNN/LSTM and transformers), the error analysis, the explainability work and the model card.

## Tech stack

Python, PyTorch, Hugging Face Transformers and Datasets, scikit-learn, TensorFlow/Keras, spaCy, NLTK, VADER, Gensim, sentence-transformers, sacrebleu, AssemblyAI, Whisper, Llama 3.3 70B. The models were trained on university GPU servers.

## Repository layout

```
src/
  speech_to_text_assemblyai.py      # transcription with AssemblyAI
  speech_to_text_whisper.py         # transcription with Whisper (small)
notebooks/
  01_nlp_features.ipynb
  02_data_quality_report.ipynb
  03_traditional_models.ipynb
  04_rnn_iterations.ipynb
  05_lstm_iterations.ipynb
  06_transformers.ipynb
  07_machine_translation.ipynb
  08_prompt_engineering_iteration_1.ipynb
  09_prompt_engineering_iterations_2_3.ipynb
  10_prompt_engineering_final.ipynb
  11_error_analysis.ipynb
  12_explainability.ipynb
  13_final_pipeline.ipynb
docs/
  model_card.md
```

## Data and models

The episode audio, transcripts, emotion labels, training datasets, model weights and predictions are not included. They contain the client's material and other people's dialogue. The notebooks expect their input files in `data/`, using the original file names, for example `data/group_23_url_1_transcript.csv` for the labelled test transcript. The final pipeline expects these model folders:

```
models/
  en_nl/                    # English -> Dutch translation model
  nl_en/                    # Dutch -> English translation model
  RoBERTa_Large_weighted/   # emotion classifier
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # RNN/LSTM notebooks: requirements-tensorflow.txt
python -m spacy download en_core_web_sm
cp .env.example .env               # AssemblyAI key and LLM endpoint
```

1. **Transcribe an episode:**

   ```bash
   python src/speech_to_text_assemblyai.py episode.mp3 --out transcribed_data_assemblyAI.xlsx
   python src/speech_to_text_whisper.py episode.mp3 --out transcribed_data_whisper.csv
   ```

2. **Explore and train:** run the notebooks in `notebooks/` in order. Training the transformers and translation models needs a GPU.
3. **Run the full pipeline:** open `notebooks/13_final_pipeline.ipynb` with the three model folders in `models/`. It writes `final_emotion_output.csv`, with one emotion per sentence.
