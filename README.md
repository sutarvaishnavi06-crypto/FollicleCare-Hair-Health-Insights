# FollicleCare — Hair & Scalp Health Insights

An interactive application for exploring hair and scalp health observations alongside a machine learning probability-calibration experiment.

## Features

- Hair and scalp symptom check-in.
- Scalp image upload with preview and observation notes.
- Logistic Regression and Random Forest model comparison.
- Probability calibration using Platt (sigmoid) scaling and isotonic regression.
- Reliability diagrams and model evaluation using ECE, Brier Score, ROC-AUC and accuracy.
- Downloadable model performance metrics.

## Installation & Setup

Clone or download this repository and open the project folder in VS Code.

Install the required dependencies:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the application:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Dataset & Model Information

The machine learning experiment uses simulated data generated with scikit-learn. The symptom score is illustrative and is not a clinical probability or diagnosis. Image uploads are supported for preview and observation only; image-based disease classification is not implemented.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly