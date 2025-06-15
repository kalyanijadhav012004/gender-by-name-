import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Auto-load dataset from a public GitHub URL
DATA_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/name_gender_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

# Load data
df = load_data()

# Show data
st.title("Gender Prediction by Name")
st.write("Sample data from the dataset:")
st.write(df.head())

# Prepare features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['name'])
y = df['gender']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"Model Accuracy: {accuracy:.2f}")

# Input for prediction
name_input = st.text_input("Enter a name to predict gender:")

if name_input:
    name_vector = vectorizer.transform([name_input])
    prediction = model.predict(name_vector)[0]
    st.info(f"Predicted Gender: **{prediction}**")
