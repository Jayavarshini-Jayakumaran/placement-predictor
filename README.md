# 🎯 Campus Placement Eligibility Predictor

> *This project is part of my self-learning journey to explore Machine Learning concepts and real-world application development.*

---

## Project Overview

This project predicts whether a student is **eligible for campus placement** based on key factors like:

- CGPA
- Skill Certifications
- Internships
- Projects

It covers the full ML workflow: **data preprocessing → exploratory data analysis (EDA) → feature engineering → model training → evaluation → an interactive web app** that gives instant predictions, probability scores, and personalized insights.

---

## ⚙️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn (Machine Learning)
- Matplotlib / Seaborn (visualization)
- Jupyter Notebook (EDA)
- Streamlit (Web App UI)

---

## Project Structure

```
.
├── data.csv                 # Raw dataset
├── features.py              # Shared feature engineering (used by training + app)
├── eda.py                    # EDA script -> generates plots/ and eda_report.md
├── eda.ipynb                  # EDA notebook (same analysis, interactive)
├── eda_report.md             # Generated EDA summary report
├── train_model.py             # Preprocessing, training, evaluation
├── app.py                    # Streamlit web app
├── metrics.txt               # Generated evaluation metrics
├── plots/                    # All generated charts (EDA + evaluation)
├── model.pkl                 # Trained model (generated)
├── scaler.pkl                # Fitted StandardScaler (generated)
└── requirements.txt
```

---

## How It Works

### 1. Data Preprocessing
- Checks for and removes duplicate rows
- Checks for and handles missing values (median imputation, if needed)
- Scales features with `StandardScaler`

### 2. Exploratory Data Analysis (EDA)
Run `python eda.py` (or open `eda.ipynb`) to generate:
- Class balance (eligible vs not eligible)
- Distribution plots for CGPA, skills, internships, projects
- CGPA vs eligibility boxplot
- Eligibility rate by skills / internships / projects
- Correlation heatmap
- Pairwise feature relationships
- A full written summary in `eda_report.md`

### 3. Feature Engineering
Implemented in `features.py` and shared between training and the app:
- **experience_score** = skills + internships + projects
- **cgpa_band** = categorical CGPA bucket (Low / Medium / High)
- **cgpa_experience_interaction** = cgpa × experience_score

### 4. Model Training & Evaluation
`train_model.py`:
- Trains a **Random Forest Classifier** on engineered, scaled features
- Evaluates on a held-out test set using:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
  - Confusion matrix
  - Classification report
- Saves evaluation plots (`plots/07_confusion_matrix.png`, `plots/08_feature_importance.png`, `plots/09_roc_curve.png`) and `metrics.txt`
- Saves the trained `model.pkl` and `scaler.pkl`

### 5. Interactive Web App
`app.py` (Streamlit):
- Sliders for CGPA, skill certifications, internships, and projects
- Applies the **same feature engineering** as training
- Predicts **Eligible / Not Eligible** with a **confidence/probability score**
- Visual probability bar
- Personalized insights:
  - CGPA band
  - Top features driving the model
  - Specific suggestions to improve eligibility chances

---

## Run the Project Locally

```bash
pip install -r requirements.txt

# 1. Explore the data
python eda.py
# or open eda.ipynb in Jupyter

# 2. Train and evaluate the model
python train_model.py

# 3. Launch the web app
streamlit run app.py
```

---

## 🚀 Future Improvements

- Add more features (resume analysis, coding skills, soft skills assessment)
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Compare multiple models (Logistic Regression, XGBoost, etc.)
- Deploy online (Streamlit Cloud / AWS)
- Add a richer interactive dashboard with historical trends

---

## 🙌 Connect with Me
- LinkedIn: [Jayavarshini Jayakumaran](https://www.linkedin.com/in/jayavarshini-jayakumaran)

## 📄 License
This project is licensed under the [MIT License](LICENSE).

<p align="center"><b>Finish what you started 💻 </b></p>
