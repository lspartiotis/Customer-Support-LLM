from transformers import pipeline, set_seed
from datasets import load_dataset
import os

# Load the dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
sample_on = False

dataset_choice = 0

# Define model paths
saved_model_path = 'model/model_trained'
saved_tokenizer_path = 'model/tokenizer_trained'

# Function to load the model and tokenizer
def load_model(model_path, tokenizer_path):
    if os.path.isdir(model_path) and os.path.isdir(tokenizer_path):
        return pipeline("text-generation", model=model_path, tokenizer=tokenizer_path)
    else:
        print(f"Error loading model from {model_path}. Falling back to default model.")
        return pipeline("text-generation", model='gpt2')

# Initialize the generators
generator_fine_tuned = load_model(saved_model_path, saved_tokenizer_path)
generator_pretrained = pipeline("text-generation", model='gpt2')

# Initialize the conversation histories
fine_tuned_history = ["You are a customer support agent. Please respond to customer queries accordingly."]
pretrained_history = ["You are a customer support agent. Please respond to customer queries accordingly."]
max_conversation_length = 6

# Define a function to generate a response using the fine-tuned model
def generate_response_fine_tuned(prompt):
    global fine_tuned_history
    
    # Append the new prompt to the conversation history
    fine_tuned_history.append(f"Customer: {prompt}\nAssistant:")
    
    # Generate a response
    conversation = "\n".join(fine_tuned_history)
    response = generator_fine_tuned(conversation, max_new_tokens=50)[0]['generated_text']
    
    # Extract only the new response and update the conversation history
    new_response = response[len(conversation):].strip()
    fine_tuned_history[-1] += " " + new_response
    
    # Truncate conversation history if it exceeds max length
    if len(fine_tuned_history) > max_conversation_length:
        fine_tuned_history = fine_tuned_history[-max_conversation_length:]
    
    return new_response

# Define a function to generate a response using the pre-trained model
def generate_response_pretrained(prompt):
    global pretrained_history
    
    # Append the new prompt to the conversation history
    pretrained_history.append(f"Customer: {prompt}\nAssistant:")
    
    # Generate a response with the pre-trained model
    conversation = "\n".join(pretrained_history)
    response = generator_pretrained(conversation, max_new_tokens=50)[0]['generated_text']
    
    # Extract only the new response and update the conversation history
    new_response = response[len(conversation):].strip()
    pretrained_history[-1] += " " + new_response
    
    # Truncate conversation history if it exceeds max length
    if len(pretrained_history) > max_conversation_length:
        pretrained_history = pretrained_history[-max_conversation_length:]
    
    return new_response

# Main loop for up to 3 queries
for _ in range(3):
    user_input = input("Customer: ")
    
    response_fine_tuned = generate_response_fine_tuned(user_input)
    response_pretrained = generate_response_pretrained(user_input)
    
    print(f"Assistant (Fine-tuned): {response_fine_tuned}")
    print(f"Assistant (Pre-trained): {response_pretrained}")

    # Exit condition (optional)
    if user_input.lower() in ["exit", "quit"]:
        break
