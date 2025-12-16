# Content-Based-Podcast-Episode-Recommendation-System

This repository provides a scalable, end-to-end pipeline for podcast episode recommendation. It supports both lexical and semantic retrieval models (e.g., TF-IDF, BERT-based embeddings) and includes a realistic evaluation framework for ranking quality.

Features

- Modular data preprocessing and feature engineering

- Lexical models (TF-IDF)

- Semantic models (Sentence-BERT)

- Scalable training and evaluation pipeline

- Reproducible experiments with saved metrics, plots, and tables

Setup
Environment

Python 3.8+

pip

It is strongly recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

Dependencies

Install all required packages using:

pip install -r requirements.txt

Core dependencies:

pandas

numpy

scikit-learn

sentence-transformers

matplotlib

seaborn

joblib

scipy

tqdm

Data

Note: Only episodeLevelDatasetSample.jsonl is included

You can obtain full public podcast episode datasets from HuggingFace. One commonly used option is SPoRC. Search HuggingFace datasets for:

SPoRC

Once downloaded, place the raw data file in the data/ directory.

How to Run

1. Data Preprocessing & Feature Engineering, use the part1_eda.ipynb

Clean the raw dataset and generate modeling features:

python scripts/preprocess.py \
  --input data/your_raw_file.jsonl \
  --output data/processed_episodes.pkl
python scripts/feature_engineering.py \
  --input data/processed_episodes.pkl
  
2. Model Training, use the part2_modeltraining.ipynb

Train baseline and supervised ranking models:

python scripts/train_models.py \
  --data data/processed_episodes.pkl \
  --output models/

Trained models will be saved to the models/ directory.

3. Evaluation

Compute ranking metrics and generate plots and tables:

python scripts/evaluate.py \
  --models models/ \
  --data data/processed_episodes.pkl \
  --output results/

All evaluation outputs (metrics, figures, and tables) will be stored in the results/ directory.
