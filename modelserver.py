import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

from train.models import NextWordLSTM, NextWordGRU

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model.eval()
# torch.set_grad_enabled(False)


def generate_ai_response(prompt):
    return 'Hello there!'
    # messages = [
    #     {"role": "system", "content": "You are a friendly and casual chatbot named Rizzlord. Respond in a conversational manner."},
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=40,
    #     do_sample=True,
    #     top_k=25,
    #     top_p=0.7,
    #     temperature=0.4
    # )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]

    # return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def predict_next_word(model, sequence, idx_to_word):
    model.eval()
    sequence = torch.tensor(sequence).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(sequence)
        predicted_idx = torch.argmax(output, dim=1).item()
    return idx_to_word[predicted_idx]

def load_model(model_type):
    global loaded_word_to_idx, loaded_model
    model_load_path = f'train/{model_type}/model.pth'
    vocab_load_path = f'train/{model_type}/vocabulary.json'

    with open(vocab_load_path, 'r') as f:
        loaded_word_to_idx = json.load(f)

    loaded_model = torch.load(model_load_path, map_location=device)
    loaded_model.eval()

@app.route('/model-type', methods=['POST'])
def choose_model():
    model_type = request.json.get('model_type')
    
    global loaded_word_to_idx, loaded_model
    model_load_path = f'train/{model_type}/model.pth'
    vocab_load_path = f'train/{model_type}/vocabulary.json'

    with open(vocab_load_path, 'r') as f:
        loaded_word_to_idx = json.load(f)

    loaded_model = torch.load(model_load_path, map_location=device)
    loaded_model.eval()
    
    print(f"Model {model_type} loaded successfully")
    
    return jsonify({'message': 'Model loaded successfully'})

def generate_text(seed_text, num_words=10):
    words = seed_text.split()
    indata = [loaded_word_to_idx.get(word.lower(), loaded_word_to_idx.get('<UNK>', 0)) for word in words]
    words.append(predict_next_word(loaded_model, indata, {v: k for k, v in loaded_word_to_idx.items()}))

    return ' '.join(words)


@app.route('/generate', methods=['POST', 'OPTIONS', 'GET'])
def autocomplete_word():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        data = request.json
        seed_text = data.get('seed_text', '')
        num_words = data.get('num_words', 1) # does nothing now
        
        generated_text = generate_text(seed_text, num_words)
        
        response = jsonify({'generated_text': generated_text})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

@app.route('/response', methods=['POST', 'OPTIONS', 'GET'])
def ai_response():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
    else:
        data = request.json
 
        generated_text = generate_ai_response(data.get('prompt'))
        
        response = jsonify({'generated_text': generated_text})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

model_type = 'GRU'
load_model(model_type)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)