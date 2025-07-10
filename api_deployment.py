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

    # Custom short message logic
    if len(message.split()) <= 3 and len(message) <= 15:
        return jsonify({
            'prediction': 'ham',
            'note': 'Manually classified as ham due to short length.'
        })

    prediction = model.predict([message])[0]
    confidence = model.predict_proba([message])[0][prediction]
    label = 'spam' if prediction == 1 else 'ham'

    return jsonify({
        'prediction': label,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
@app.route('/')
def home():
    return "✅ Hybrid Spam Detection API is running."
@app.route('/health')
def health():
    return jsonify({"status": "API is running"})

