import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Gender Prediction Based on Name")

# Load data
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Show available columns
    st.write("Columns in the dataset:", df.columns.tolist())

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if 'name' in df.columns and 'gender' in df.columns:
        # Vectorize names
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['name'].astype(str))
        y = df['gender']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("Model Accuracy")
        st.write(f"{accuracy * 100:.2f}%")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
        st.pyplot(fig)

        st.subheader("Try It Yourself")
        name_input = st.text_input("Enter a name to predict gender:")
        if name_input:
            input_vec = vectorizer.transform([name_input])
            prediction = model.predict(input_vec)[0]
            st.success(f"Predicted Gender: **{prediction}**")
    else:
        st.error("Required columns 'name' and 'gender' not found. Please check the column names in your file.")