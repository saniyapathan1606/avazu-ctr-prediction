🚀 Avazu Ad Click Prediction — Machine Learning + Streamlit UI
🧭 Overview

This repository predicts whether a mobile ad will be clicked using the Avazu dataset.
It combines a trained XGBoost classifier (backend) with an interactive Streamlit app (frontend).

The app provides:

A clean, beginner-friendly UI

Predictive insights (Clicked / Not Clicked)

Model stats, metrics, and feature-based predictions

⚙️ VS Code: Setup & Workflow
1️⃣ Clone Repository
git clone https://github.com/<your-username>/Avazu_click.git
cd Avazu_click

2️⃣ Create Virtual Environment
🪟 On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

🪟 On Windows (cmd)
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt

🐧 On Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

3️⃣ Train or Use Existing Model
Option A — Train Locally (in Notebook)

Open backend/model_train.ipynb

Update the dataset path:

data = pd.read_csv("train.gz", compression="gzip")


Run all cells to:

Clean data

Encode features

Train XGBoost model

Save model + encoders

Files saved in backend/:

model.pkl
encoders.pkl

Option B — Use Existing Trained Model

If already trained elsewhere (e.g. Colab):

Copy your model.pkl and encoders.pkl into backend/.

4️⃣ Run the Streamlit Frontend

From project root:

cd frontend
streamlit run streamlit_app.py


Then open in browser:
👉 http://localhost:8501

🧠 Model Summary
Component	Description
Algorithm	XGBoost Classifier
Target	click (1 = Clicked, 0 = Not Clicked)
Input Features	Device type, connection, app category, site category, campaign type, etc.
Metric Used	Log Loss, ROC AUC Score
Goal	Predict probability of user clicking an ad
📊 Model Performance
Metric	Description	Example Value
AUC (ROC-AUC)	Measures model’s ability to separate classes (1 = perfect)	0.92
Log Loss	Penalizes incorrect probabilities	0.36
Accuracy	Correct predictions ratio	89%
🧩 Example Input & Output
Input Field	Example	Meaning
Campaign Type	"Mobile Banner"	Type of ad campaign
Device Type	"Smartphone"	User device
Connection Type	"Wi-Fi"	Network type
App Category	"Gaming"	App context
Site Category	"Entertainment"	Ad placement type

✅ Predicted Output:

“🟢 Clicked — High chance user interacts with ad.”

📦 Dependencies

Main libraries:

xgboost
pandas
numpy
scikit-learn
joblib
streamlit
matplotlib

🧹 .gitignore (important)

Add this in .gitignore to prevent uploading large data:

# Ignore heavy dataset and artifacts
train.gz
*.pkl
*.joblib
*.csv
*.zip

# Ignore environment folders
.venv/
__pycache__/
.ipynb_checkpoints/

# Ignore IDE settings
.vscode/


🧠 Interview Key Points
Topic	Explanation
Why XGBoost?	Handles large tabular data efficiently and reduces overfitting.
Why ROC-AUC?	Evaluates true positive vs false positive rate — best for imbalanced datasets.
Why Log Loss?	Measures how confident the model is in predictions.
Feature Encoding	Label encoding for categorical fields (app, site, device).
Deployment	Streamlit frontend allows quick demo without Flask or heavy API setup.
💡 Future Improvements

Use LightGBM for faster training.

Add SHAP feature explainability.

Deploy on Render or Hugging Face Spaces.
