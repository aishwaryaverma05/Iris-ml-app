import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="Iris Classifier 🌸", layout="centered")
st.title("🌸 Iris Flower Species Classifier")
st.markdown("#### Enter flower measurements to predict the species")

# Input form
with st.form("iris_form"):
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    submitted = st.form_submit_button("Predict Species 🌼")

# Prediction output
if submitted:
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=iris.feature_names)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.success(f"🌼 Predicted Species: **{iris.target_names[prediction].capitalize()}**")
    st.markdown("#### 🌡️ Prediction Probabilities")
    prob_df = pd.DataFrame({'Species': iris.target_names, 'Probability': proba})
    st.bar_chart(prob_df.set_index('Species'))

# Metrics section
with st.expander("🔍 Model Evaluation Metrics"):
    st.markdown(f"**Model Accuracy:** `{accuracy * 100:.2f}%`")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification report
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=iris.target_names))


