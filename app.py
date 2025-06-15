import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Sample dataset defined directly in code
data = {
    "name": [
        "John", "Emily", "Michael", "Sarah", "David", "Jessica", "Daniel", "Ashley",
        "Matthew", "Amanda", "Chris", "Brittany", "Joshua", "Megan", "Andrew", "Lauren",
        "Joseph", "Stephanie", "Justin", "Nicole", "Brian", "Heather", "James", "Elizabeth"
    ],
    "gender": [
        "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Streamlit UI
st.title("Gender Prediction by Name (No File Upload Needed)")
st.write("This app predicts gender based on the name using a simple machine learning model.")

# Show dataset
st.subheader("Sample Data")
st.dataframe(df)

# Prepare data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['name'])
y = df['gender']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Show model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"Model Accuracy on Test Data: {accuracy:.2f}")

# Prediction input
st.subheader("Predict Gender from Name")
name_input = st.text_input("Enter a name:")

if name_input:
    input_vector = vectorizer.transform([name_input])
    prediction = model.predict(input_vector)[0]
    st.info(f"Predicted Gender: **{prediction}**")

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
