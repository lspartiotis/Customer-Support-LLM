from transformers import GPT2Tokenizer
import torch
from data import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#tokenizer.padding_side = "left" ;# for batch generation
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function_pair(examples):
    return tokenizer(examples['dialogue_pair'], truncation=True, padding='max_length', return_tensors='pt').to(device)

tokenized_dataset = dataset['train'].map(preprocess_function_pair, batched=True)


#decoded_text = tokenizer.decode(tokenized_dataset['tokenized_text_x'][0]['input_ids'], skip_special_tokens=True)

### test it
decoded_text = tokenizer.decode(tokenized_dataset['input_ids'][0], skip_special_tokens=True)
print(decoded_text)


### save tokenized dataset locally ####
from datasets import load_from_disk

tok_name = "tokenized_dataset"
tokenized_dataset.save_to_disk(tok_name)

saved_tokenized_path = f'{tok_name}'

### load tokenized dataset ####
from datasets import load_from_disk
tokenized_dataset = load_from_disk(saved_tokenized_path)


print(tokenized_dataset.features)


### Final formalizing
tokenized_dataset = tokenized_dataset.remove_columns(["flags", "category","intent", "instruction", "response", "cleaned_text_x", "cleaned_text_y"])
tokenized_dataset.set_format("torch")

# Split the dataset

split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(train_dataset)