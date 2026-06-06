# 🎬 Netflix Customer Churn Prediction Dashboard

An interactive Machine Learning web application designed to predict customer churn status for Netflix subscribers. This project transitions exploratory data analysis and predictive models from a Google Colab environment into a production-ready, minimalist web dashboard built using **Streamlit**.

🌐 **Live Application URL:** https://netflixchurnprediction-mauryaji.streamlit.app/

---

## 📱 Dashboard Interface Overview
![Project Banner](netflix.png)  
*(Note: If your downloaded photo extension is .jpg or .jpeg, please change the code line above to netflix.jpg or netflix.jpeg)*

---

## 📊 Project Details & Core Analytics

This project involves an end-to-end predictive framework that processes customer behavior metrics to forecast retention risks.

### 1. Data Pipeline & Engineering
- **Target Mapping:** Binary classification alignment where `Yes` (User Churns) is mapped to `1` and `No` (User stays retained) is mapped to `0`.
- **Feature Encoding:** High-cardinality categorical attributes (such as *Device Used Most Often*, *Genre Preference*, *Region*, *Payment History*, and *Subscription Plan*) are transformed via One-Hot Encoding (`pd.get_dummies(..., drop_first=True)`).
- **Dimensionality Trimming:** Irrelevant identifier matrices like `Customer ID` are programmatically dropped to filter out noise and improve algorithmic speed.

### 2. Algorithmic Configurations
The backend model relies strictly on the mathematical hyperparameters trained within the exploratory notebook:
- **Primary Architecture:** Decision Tree Classifier (`max_depth=5`, `random_state=42`)
- **Validation Splitting:** 80-20 Train-Test split configuration (`test_size=0.2`).
- **Optimization Strategy:** Frontend features restrict user inputs to the Top 5 core structural attributes (*Customer Satisfaction Score*, *Engagement Rate*, *Daily Watch Time*, *Subscription Length*, and *Support Queries*). Lower-weighted indicators are imputed dynamically in the background using statistical medians to maintain structural matrix compliance.

### 3. Model Benchmark Analysis
During exploratory research, the dataset was benchmarked across multi-layered trees:
- **Decision Tree Classifier:** Evaluated with customized leaf partitions to capture maximum structural information gain.
- **Random Forest Ensemble:** Used as a multi-estimator baseline (`n_estimators=200`) to extract definitive feature importance distribution arrays.

---

## ⚡ Core ML Concepts Applied
- **Feature Weights Alignment:** Rather than processing arbitrary metrics, it evaluates parameters using real-time feature importance splits (`dt_model.feature_importances_`).
- **Dynamic Array Validation:** Ensures that arbitrary test cases are shaped perfectly through schema alignment filters (`input_df[X.columns]`) prior to passing data arrays to the predictor.
- **Confidence Calibration:** Uses dynamic mapping arrays (`predict_proba`) to yield percentage-based certainty indicators instead of standard rigid binary flags.

---

