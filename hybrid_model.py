import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from enron_preprocessing import load_enron_dataset, preprocess_and_split

# Load and preprocess
df = load_enron_dataset("enron/spam", "enron/ham")
X_train, X_test, y_train, y_test = preprocess_and_split(df)

# Define base classifiers
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(probability=True, kernel='linear'))
])

# Combine in ensemble
ensemble = VotingClassifier(
    estimators=[
        ('nb', nb_pipeline),
        ('svm', svm_pipeline)
    ],
    voting='soft'
)

# Train
print("Training started...")
ensemble.fit(X_train, y_train)
print("Training finished!")

# Evaluate
y_pred = ensemble.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Hybrid Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
joblib.dump(ensemble, 'hybrid_spam_model.pkl')
