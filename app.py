import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Page configurations
st.set_page_config(page_title="Netflix Churn Predictor", layout="centered")

st.title("🎬 Netflix Customer Churn Prediction App")
st.write("Enter the customer details manually using the inputs on the left. These are the top features your Colab model uses to predict churn.")

# --- STEP 1: BACKGROUND DATA LOAD ---
@st.cache_data
def load_data():
    df = pd.read_excel("netflix_large_user_data.xlsx")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ 'netflix_large_user_data.xlsx' file was not found! Please copy it to your project folder.")
    st.stop()

# --- STEP 2: DATA PREPROCESSING (100% Match with Colab) ---
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

# --- STEP 3: MODEL TRAINING (Exact Colab Hyperparameters) ---
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, Y_train)

# --- STEP 4: SIDEBAR INPUT FIELDS (Actual Top Impact Features Fix) ---
st.sidebar.header("📊 Top Model Features")
st.sidebar.write("Change these to see different prediction results:")

# Replacing 0% importance features with your Colab model's highest-ranking features
income = st.sidebar.number_input("Monthly Income ($)", min_value=500, max_value=15000, value=5000, step=500)
age = st.sidebar.slider("Age", 18, 70, 35)
watch_time = st.sidebar.slider("Daily Watch Time (Hours)", 0.5, 5.0, 2.5)
support_queries = st.sidebar.slider("Support Queries Logged", 0, 10, 3)
profiles = st.sidebar.slider("Number of Profiles Created", 1, 5, 2)

# Background fill for remaining less important columns
input_data = {col: X[col].median() for col in X.columns}

# Overwriting with your actual dynamic high-importance inputs
input_data['Monthly Income ($)'] = income
input_data['Age'] = age
input_data['Daily Watch Time (Hours)'] = watch_time
input_data['Support Queries Logged'] = support_queries
input_data['Number of Profiles Created'] = profiles

input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]  # Keep identical column order

# --- STEP 5: MAIN SCREEN PREDICTION ---
st.subheader("🎯 Make a Prediction")
st.write("Adjust the values in the sidebar and click the button below to see how prediction shifts.")

if st.button("Predict Churn Status", use_container_width=True):
    prediction = dt_model.predict(input_df)
    prediction_proba = dt_model.predict_proba(input_df)
    
    st.write("---")
    if prediction[0] == 1:
        st.error(f"🔴 Prediction: This user is likely to Churn (Cancel Subscription). Confidence: {prediction_proba[0][1]:.2%}")
    else:
        st.success(f"🟢 Prediction: This user is likely to Stay Active (Retained). Confidence: {prediction_proba[0][0]:.2%}")
    st.write("---")
