from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Load the models
saved_model_path = 'model/model_trained'
saved_tokenizer_path = 'model/tokenizer_trained'

def load_model(model_path, tokenizer_path):
    if os.path.isdir(model_path) and os.path.isdir(tokenizer_path):
        return pipeline("text-generation", model=model_path, tokenizer=tokenizer_path)
    else:
        print(f"Error loading model from {model_path}. Falling back to default model.")
        return pipeline("text-generation", model='gpt2')

generator_fine_tuned = load_model(saved_model_path, saved_tokenizer_path)
generator_pretrained = pipeline("text-generation", model='gpt2')

fine_tuned_history = ["You are a customer support agent. Please respond to customer queries accordingly."]
pretrained_history = ["You are a customer support agent. Please respond to customer queries accordingly."]
max_conversation_length = 6

def generate_response(generator, history, prompt):
    history.append(f"Customer: {prompt}\nAssistant:")
    conversation = "\n".join(history)
    response = generator(conversation, max_new_tokens=50)[0]['generated_text']
    new_response = response[len(conversation):].strip()
    history[-1] += " " + new_response
    
    if len(history) > max_conversation_length:
        history = history[-max_conversation_length:]
    
    return new_response, history

@app.route('/chat', methods=['POST'])
def chat():
    global fine_tuned_history, pretrained_history

    data = request.json
    user_input = data.get('message', '')
    model_type = data.get('model', 'fine_tuned')
    
    if model_type == 'fine_tuned':
        response, fine_tuned_history = generate_response(generator_fine_tuned, fine_tuned_history, user_input)
    else:
        response, pretrained_history = generate_response(generator_pretrained, pretrained_history, user_input)
    
    return jsonify({"response": response})

@app.route('/reset', methods=['POST'])
def reset():
    global fine_tuned_history, pretrained_history
    fine_tuned_history = ["You are a customer support agent. Please respond to customer queries accordingly."]
    pretrained_history = ["You are a customer support agent. Please respond to customer queries accordingly."]
    return jsonify({"message": "Histories reset."})

if __name__ == '__main__':
    app.run(debug=True)
