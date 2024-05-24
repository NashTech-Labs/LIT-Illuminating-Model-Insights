from flask import Flask, request, jsonify,make_response
import mlrun
from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
import torch
from utils.constants import serving_model_path
from utils.helpers import load_project
import pickle
app = Flask(__name__)



@mlrun.handler()
def predict_sentiment(texts,tokenizer,model):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")  # Tokenize input text
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        sentiment_map_inst = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment_class = sentiment_map_inst.get(predicted_class, '')

        results.append({"text":text,"sentiment":sentiment_class})
    return results


@app.route('/predict', methods=['POST'])
def predict():
    """
    Model serving component serves local flask app to the mlrun ui
    """
    project=load_project()
    context = mlrun.get_or_create_ctx("transformers-example")
    if request.method == 'POST':
        data = request.get_json()
        texts = data.get('text')
        config_file=AutoConfig.from_pretrained(f'{serving_model_path}/')
        model = AutoModelForSequenceClassification.from_pretrained(f'{serving_model_path}/',config=config_file)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=f'{serving_model_path}')
        results = predict_sentiment(texts, tokenizer, model)
        context.log_result(key='Result:',value=results)
        return make_response(jsonify(results), 200)



app.run(port=8004)