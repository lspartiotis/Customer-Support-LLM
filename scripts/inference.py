from transformers import pipeline, set_seed
from tokenizer import *


saved_model_path = 'model/model_trained'
saved_tokenizer_path = 'model/tokenizer_trained'





### Load the dataset from huggingface
from datasets import load_dataset

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
sample_on = False


dataset_choice = 0


fuego1 = ""

prompt = input(fuego1)
#prompt = "\nCustomer: Hello I would like to cancel my order. \nAssistant:"
#prompt = "\nCustomer: " + eval_dataset['cleaned_text_x'][0] + tokenizer.eos_token + "\nAssistant: "
#set_seed(42)
generator = pipeline("text-generation", model=saved_model_path, tokenizer=saved_tokenizer_path)
print(generator(prompt, max_length=100))

#### same prompt to pretrained unfinetuned model:
generator = pipeline("text-generation", model='gpt2')
response = generator(prompt)
print(response)
