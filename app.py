import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Page configurations
st.set_page_config(page_title="Netflix Churn Predictor", layout="centered")

st.title("🎬 Netflix Customer Churn Prediction App")
st.write("This app predicts Netflix customer churn using only the most important features from your Colab model.")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_excel("netflix_large_user_data.xlsx")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ 'netflix_large_user_data.xlsx' file was not found! Please copy it to your project folder.")
    st.stop()

# Data Preprocessing
df_model = df.copy()
df_model['Churn Status (Yes/No)'] = df_model['Churn Status (Yes/No)'].map({'Yes': 1, 'No': 0})
categorical_cols = [
    'Device Used Most Often',
    'Genre Preference',
    'Region',
    'Payment History (On-Time/Delayed)',
    'Subscription Plan'
]

df_processed = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
if 'Customer ID' in df_processed.columns:
    df_processed = df_processed.drop('Customer ID', axis=1)

X = df_processed.drop('Churn Status (Yes/No)', axis=1)
Y = df_processed['Churn Status (Yes/No)']

# Train Model to extract Feature Importances
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, Y_train)

# --- SIDEBAR: ONLY TOP IMPORTANT INPUTS ---
st.sidebar.header("📊 Top Important Features")
st.sidebar.write("Only showing features that impact prediction the most:")

# Top 5 core numerical features that matter most
sat_score = st.sidebar.slider("Customer Satisfaction Score (1-10)", 1, 10, 5)
engagement = st.sidebar.slider("Engagement Rate (1-10)", 1, 10, 5)
watch_time = st.sidebar.slider("Daily Watch Time (Hours)", 0.5, 5.0, 2.5)
sub_length = st.sidebar.slider("Subscription Length (Months)", 1, 24, 12)
support_queries = st.sidebar.slider("Support Queries Logged", 0, 10, 3)

# Process only these inputs, set rest to median/default to keep code stable
input_data = {col: X[col].median() for col in X.columns} # automatic fallback for low impact features

input_data['Customer Satisfaction Score (1-10)'] = sat_score
input_data['Engagement Rate (1-10)'] = engagement
input_data['Daily Watch Time (Hours)'] = watch_time
input_data['Subscription Length (Months)'] = sub_length
input_data['Support Queries Logged'] = support_queries

input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]

# --- MAIN SCREEN: PREDICTION ---
st.subheader("🎯 Make a Prediction")
if st.button("Predict Churn Status", use_container_width=True):
    prediction = dt_model.predict(input_df)
    prediction_proba = dt_model.predict_proba(input_df)
    
    st.write("---")
    if prediction[0] == 1:
        st.error(f"🔴 Prediction: This user is likely to Churn. Confidence: {prediction_proba[0][1]:.2%}")
    else:
        st.success(f"🟢 Prediction: This user is likely to Stay Active. Confidence: {prediction_proba[0][0]:.2%}")
    st.write("---")

# --- MAIN SCREEN: FEATURE IMPORTANCE PLOT ---
st.subheader("📊 Model Feature Importance")
st.write("This plot shows which features your Decision Tree model used the most (Highest to Lowest impact).")

# Calculate importances
importances = dt_model.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(7)

# Plotting
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=feat_importances.values, y=feat_importances.index, palette="viridis", ax=ax)
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
plt.tight_layout()

st.pyplot(fig)