from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = classifier(text)[0]
    label = result['label']
    score = result['score']

    if label == "NEUTRAL" or score < 0.7:
        bias_result = "Not Biased"
    else:
        bias_result = "Biased"

    return jsonify({
        "input": text,
        "bias": bias_result,
        "sentiment_label": label,
        "confidence": round(score, 3)
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
