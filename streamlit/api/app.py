from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)

model_name = "tuner007/pegasus_summarizer"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer_summarizer = PegasusTokenizer.from_pretrained(model_name)
model_summarizer = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    tldr = get_response(text)
    return jsonify({'tldr': tldr})

def get_response(input_text):
    batch = tokenizer_summarizer(
        [input_text],
        truncation=True,
        padding="longest",
        max_length=1024,
        return_tensors="pt",
    ).to(torch_device)
    gen_out = model_summarizer.generate(
        **batch, max_length=256, num_beams=5, num_return_sequences=1, temperature=1.5
    )
    output_text = tokenizer_summarizer.batch_decode(gen_out, skip_special_tokens=True)
    return output_text[0]

if __name__ == '__main__':
    app.run(debug=False, port=5001)
