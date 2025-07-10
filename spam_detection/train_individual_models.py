# =============================
# train_individual_models.py
# =============================

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Enron-Spam structure
def load_data():
    data = []
    for label, folder in [(0, 'enron/ham'), (1, 'enron/spam')]:
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r', encoding='latin-1') as f:
                data.append((f.read(), label))
    return pd.DataFrame(data, columns=['text', 'label'])

# Load data
df = load_data()

# Preprocessing
X = df['text']
y = df['label']
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Train SVM
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluation
print("\nNaive Bayes Report:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix (NB):")
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d')
plt.title("Naive Bayes Confusion Matrix")
plt.show()

print("\nSVM Report:")
print(classification_report(y_test, y_pred_svm))
print("Confusion Matrix (SVM):")
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d')
plt.title("SVM Confusion Matrix")
plt.show()

# Save models and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(nb_model, "models/naive_bayes_model.pkl")
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
