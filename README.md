# 📧 Hybrid Spam Detection API (Naïve Bayes + SVM)

This is a Flask-based RESTful API for real-time email spam detection using a hybrid machine learning model that combines **Naïve Bayes** and **Support Vector Machine (SVM)** classifiers. The model is trained on the **Enron-Spam dataset** and deployed for live predictions via HTTP requests.

---

## 🚀 Features

- ✅ Real-time spam prediction API
- ✅ Uses a hybrid of Naïve Bayes and SVM
- ✅ Trained on 35,000+ emails from the Enron dataset
- ✅ Deployable via platforms like Render or Railway
- ✅ Includes CORS support for frontend integration

---

## 🧠 Model Architecture

- **Vectorizer:** TF-IDF
- **Classifier 1:** Multinomial Naïve Bayes
- **Classifier 2:** Support Vector Machine (Linear Kernel)
- **Hybrid Logic:** Majority voting of NB + SVM predictions

---

## 📁 Project Structure

