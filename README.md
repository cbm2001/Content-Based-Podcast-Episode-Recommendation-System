# Content-Based-Podcast-Episode-Recommendation-System
Setup
Environment:
Requires Python 3.8+ and pip. Recommended to use a virtual environment (e.g., venv or conda).

Dependencies:
Install required packages via:

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

Note: The dataset is not included due to licensing restrictions. Dataset can be found on HuggingFace by searching SPoRC.

How to Run

Data Preprocessing & Feature Engineering

Run preprocessing to clean raw data and create modeling features.

python scripts/preprocess.py --input data/your_raw_file.jsonl --output data/processed_episodes.pkl
python scripts/feature_engineering.py --input data/processed_episodes.pkl
Model Training

Train baseline and supervised ranking models.

python scripts/train_models.py --data data/processed_episodes.pkl --output models/

Evaluation

Run evaluation to compute metrics and generate figures/tables.

python scripts/evaluate.py --models models/ --data data/processed_episodes.pkl --output results/
All output metrics, plots, and tables will be saved in the results/ directory.
