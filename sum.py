from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# مدل خلاصه‌سازی mt5-small
summarizer = pipeline(
    "summarization",
    model="HooshvareLab/pn-summary-mt5-small",
    tokenizer="HooshvareLab/pn-summary-mt5-small"
)

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return jsonify({"summary": result[0]['summary_text']})

@app.route('/', methods=['GET'])
def home():
    return "MT5-small Summarizer is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)