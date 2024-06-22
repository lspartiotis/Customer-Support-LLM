from transformers import pipeline, set_seed
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
sample_on = False

dataset_choice = 0

# Load the fine-tuned model and tokenizer
saved_model_path = 'model/model_trained'
saved_tokenizer_path = 'model/tokenizer_trained'

# Initialize the conversation histories
fine_tuned_history = ""
pretrained_history = ""

# Define a function to generate a response using the fine-tuned model
def generate_response_fine_tuned(prompt):
    global fine_tuned_history
    
    # Append the new prompt to the conversation history
    fine_tuned_history += f"\nCustomer: {prompt}\nAssistant:"
    
    # Generate a response
    generator = pipeline("text-generation", model=saved_model_path, tokenizer=saved_tokenizer_path)
    response = generator(fine_tuned_history, max_length=100)[0]['generated_text']
    
    # Extract only the new response and update the conversation history
    new_response = response[len(fine_tuned_history):]
    fine_tuned_history += new_response
    
    return new_response.strip()

# Define a function to generate a response using the pre-trained model
def generate_response_pretrained(prompt):
    global pretrained_history
    
    # Append the new prompt to the conversation history
    pretrained_history += f"\nCustomer: {prompt}\nAssistant:"
    
    # Generate a response with the pre-trained model
    generator = pipeline("text-generation", model='gpt2')
    response = generator(pretrained_history, max_length=100)[0]['generated_text']
    
    # Extract only the new response and update the conversation history
    new_response = response[len(pretrained_history):]
    pretrained_history += new_response
    
    return new_response.strip()

# Main loop for continuous conversation
while True:
    user_input = input("Customer: ")
    
    response_fine_tuned = generate_response_fine_tuned(user_input)
    response_pretrained = generate_response_pretrained(user_input)
    
    print(f"Assistant (Fine-tuned): {response_fine_tuned}")
    print(f"Assistant (Pre-trained): {response_pretrained}")

    # Exit condition (optional)
    if user_input.lower() in ["exit", "quit"]:
        break
