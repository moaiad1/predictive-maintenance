# Predictive Maintenance for Industrial Equipment
**Using Multi-Model Machine Learning on the AI4I 2020 Dataset**

## Project Structure
```
Data_mining/
├── data/
│   ├── raw/           # Original AI4I 2020 dataset
│   └── processed/     # Cleaned & feature-engineered data
├── notebooks/         # Jupyter notebooks for exploration & analysis
├── src/               # Python source modules
│   ├── preprocess.py  # Data cleaning, feature engineering, SMOTE
│   ├── train.py       # Model training & hyperparameter tuning
│   └── evaluate.py    # Metrics, plots, confusion matrices
├── models/            # Saved trained models (.pkl)
├── results/
│   ├── figures/       # ROC curves, confusion matrices, feature importance
│   └── metrics/       # CSV/JSON with evaluation scores
└── app/               # Streamlit monitoring application
    └── app.py
```

## Models
- Random Forest (RF)
- Support Vector Machine (SVM with RBF kernel)
- Logistic Regression (baseline)

## Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, MCC

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Run full pipeline
python src/train.py

# Launch monitoring app
streamlit run app/app.py
```
