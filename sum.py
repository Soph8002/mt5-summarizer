from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

# بارگذاری مدل و توکنایزر
tokenizer = T5Tokenizer.from_pretrained("nafisehNik/mt5-persian-summary")
model = T5ForConditionalGeneration.from_pretrained("nafisehNik/mt5-persian-summary")

app = Flask(__name__)

@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # تبدیل به توکن
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # تولید خلاصه
    summary_ids = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)