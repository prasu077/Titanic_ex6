import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

# ----------------------------------
# App Title
# ----------------------------------
st.title("🚢 Titanic Survival Prediction")
st.write("Gaussian Naive Bayes Classification Model")

# ----------------------------------
# Load Dataset
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("titanic.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------------------------
# Data Preprocessing
# ----------------------------------
st.subheader("Data Preprocessing")

# Drop unnecessary columns
df = df.drop(
    ['passenger_id', 'name', 'sib_sp', 'parch', 'ticket', 'cabin', 'embarked'],
    axis=1
)

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)

# One-hot encoding for sex
df = pd.get_dummies(df, columns=['sex'], drop_first=True)

# Standardize fare
scaler = StandardScaler()
df[['fare']] = scaler.fit_transform(df[['fare']])

st.write("Processed Dataset")
st.dataframe(df.head())

# ----------------------------------
# Feature Selection
# ----------------------------------
X = df[['p_class', 'age', 'fare', 'sex_male']]
y = df['survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ----------------------------------
# Model Training
# ----------------------------------
st.subheader("Model Training")

model = GaussianNB()
model.fit(X_train, y_train)

st.success("Gaussian Naive Bayes model trained successfully!")

# ----------------------------------
# Predictions
# ----------------------------------
y_pred = model.predict(X_test)

# ----------------------------------
# Evaluation
# ----------------------------------
st.subheader("Model Evaluation")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# ----------------------------------
# User Input Prediction
# ----------------------------------
st.subheader("Predict Survival for a Passenger")

p_class = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0.0, value=30.0)

sex_male = 1 if sex == "Male" else 0
fare_scaled = scaler.transform([[fare]])[0][0]

input_data = np.array([[p_class, age, fare_scaled, sex_male]])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")
