# Cell 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cell 2: Load the dataset
df = pd.read_csv("name_gender_dataset.csv")
df.head()
# Cell 3: Check for missing values and basic info
print(df.info())
print(df.isnull().sum())
df['gender'].value_counts()
# Cell 4: Preprocessing - Convert text to lowercase and drop missing data
df['name'] = df['name'].str.lower()
df.dropna(inplace=True)
# Cell 5: Feature extraction using CountVectorizer (character-level)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
X = vectorizer.fit_transform(df['name'])
y = df['gender']
# Cell 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Cell 7: Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Cell 8: Make predictions and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Cell 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# Cell 10: Predict gender from a custom name
def predict_gender(name):
    name = name.lower()
    name_vec = vectorizer.transform([name])
    return model.predict(name_vec)[0]

# Example
predict_gender("Kalyani")
