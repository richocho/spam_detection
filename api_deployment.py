from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # ✅ This line allows requests from your frontend

model = joblib.load("hybrid_spam_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    spam_keywords = ['won', 'free', 'congratulations', 'money', 'prize', 'urgent', 'click here', 'offer']

    # Short message filter — only if it's not obviously spam
    if len(message.split()) <= 5 and len(message) <= 30:
        if any(word in message.lower() for word in spam_keywords):
            pass  # let model decide
        else:
            return jsonify({
                'prediction': 'ham',
                'note': 'Classified as ham due to short and simple content.'
            })

    prediction = model.predict([message])[0]
    confidence = model.predict_proba([message])[0][prediction]
    label = 'spam' if prediction == 1 else 'ham'

    return jsonify({
        'prediction': label,
        'confidence': round(confidence * 100, 2)
    })

    
@app.route('/')
def home():
    return "✅ Hybrid Spam Detection API is running."
@app.route('/health')
def health():
    return jsonify({"status": "API is running"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

