# JPENCorpLex: Multilingual Sentence Alignment for Financial Transcripts

## Overview
**JPENCorpLex** is a learning project aimed at developing a system that aligns Japanese and English sentences from financial transcripts. This initiative serves as a practical exploration of Natural Language Processing (NLP) and the use of transformer libraries, particularly `spaCy` and Hugging Face's `transformers`. 

## Project Purpose
The primary goal of this project is to facilitate the understanding of multilingual financial communications. By aligning sentences from Japanese and English transcripts, the project aims to provide accessible insights for analysts and researchers navigating language barriers in the global finance landscape.

## Key Approach
1. **Data Extraction and Cleaning:** Financial transcripts are extracted from PDFs and cleaned to remove unnecessary information such as headers and speaker labels.
2. **Sentence Segmentation:** Using `spaCy` for English and `MeCab` for Japanese, sentences are segmented for alignment.
3. **Model Training:** The project utilizes the `SentenceTransformer` model (`paraphrase-multilingual-MiniLM-L12-v2`) to encode sentences and calculate similarity scores.
4. **Alignment Process:** Sentences are aligned based on a predefined cosine similarity threshold, allowing for the identification of matching pairs.

## Applications
- **Financial Analysis:** Enables analysts to compare corporate strategies presented in different languages.
- **Multilingual Communication:** Assists organizations in communicating effectively across language barriers.
- **Academic Research:** Provides aligned datasets for studies in comparative finance or sentiment analysis.

## Caveats
- This project is a work in progress, as I am a non-coder aiming to familiarize myself with NLP and transformer libraries. 
- The codebase requires significant cleanup and optimization to reach production quality. The current implementation is a reflection of my learning journey, and I welcome contributions and suggestions for improvement.

## Next Steps
- Explore additional datasets to enhance model performance.
- Experiment with various sentence embedding models for comparative analysis.
- Consider developing a web interface to allow users to input new transcripts for alignment and analysis.
