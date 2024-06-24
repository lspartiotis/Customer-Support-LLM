from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

##### After fine-tuning our model:
model_trained_path = "/content/drive/MyDrive/ai_unipi/Deep Learning/model_trained"
# 1. load a pretrained model
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_trained_path) ### finetuned
#model_ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
model_ref = create_reference_model(model)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

peft_training = False
if peft_training:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",

    )
    peft_config = lora_config

    load model in 8bit
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_trained_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, # for 4Bit quantization
        trust_remote_code=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        )
        #    peft_config=lora_config,

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)





import torch
import warnings
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, Trainer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
from sklearn.model_selection import train_test_split
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


tqdm.pandas()

bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

## model config ##
model_config = ModelConfig(
    model_name_or_path="gpt2",
    attn_implementation=None, # or "flash_attention_2"
    torch_dtype=torch.bfloat16,
)
###
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)


config = RewardConfig(
    output_dir="reward_modeling",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    learning_rate=1.41e-5,
    remove_unused_columns=False,
    optim="adamw_torch",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    max_length=512
)

model_id = "gpt2"
################
# Model & Tokenizer
################

quantization_config = get_quantization_config(model_config)
model_kwargs = dict(
    trust_remote_code=True,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=1, **model_kwargs
    ).to(device)
model = prepare_model_for_kbit_training(model) #####custom


tokenizer.pad_token = tokenizer.eos_token

if peft_config.task_type != "SEQ_CLS":
    warnings.warn(
        "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
        " Make sure to pass --lora_task_type SEQ_CLS when using this script."
)

################
# Dataset
################
#raw_datasets = load_dataset("Anthropic/hh-rlhf")
raw_datasets = merged_dataset # from the Attempt section

# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples
def my_preprocess_function(examples):

    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples
def my_concat_preprocess_funtion (examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    i = 0
    for query, chosen, rejected in zip(examples['query'], examples["chosen"], examples["rejected"]):
        #print("\nCH::: \n", chosen)
        #print( "\nREJ::", i , rejected)
        chosen = f"\nCustomer: {query} " + tokenizer.eos_token + "\nAssistant: " + chosen
        rejected = f"\nCustomer: {query} " + tokenizer.eos_token + "\nAssistant: " + rejected
        tokenized_chosen = tokenizer(chosen, padding="max_length", truncation=True, return_tensors='pt').to(device)
        tokenized_rejected = tokenizer(rejected, padding="max_length", truncation=True, return_tensors='pt').to(device)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        i += 1

    return new_examples
# Preprocess the dataset and filter out examples that are longer than args.max_length
#for entry in raw_datasets:
#    print(entry)
print(raw_datasets)
raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)#my_concat_preprocess_funbction
print(raw_datasets[0])
raw_datasets = raw_datasets.filter(
    lambda x: len(x["input_ids_chosen"]) <= 512 and len(x["input_ids_rejected"]) <= 512
)

print(raw_datasets[0])
split_dataset = raw_datasets.train_test_split(test_size=0.2)
print(split_dataset)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']
print("Dataset features:", train_dataset.features)
print("Number of rows:", train_dataset.num_rows)

#train_dataset = raw_datasets["train"].select(range(2))
#eval_dataset = raw_datasets["test"]

################
# Training
################

tokenizer.pad_token = tokenizer.eos_token

class mRewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_chosen"],  attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_k = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

peft_training = False
if peft_training:
    peft_model=get_peft_model(model, peft_config)

    trainer = RewardTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset = train_dataset,## my tokenized data from Attempt section
        args=config,
        peft_config=peft_config,
        eval_dataset=eval_dataset,
        )
else:
    peft_model=get_peft_model(model, peft_config)

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset = train_dataset,## my tokenized data from Attempt section
        args=config,
        eval_dataset=eval_dataset,
    )

print(train_dataset)
#trainer.compute_loss(
#   model=model,
#   inputs = train_dataset)

#tokenized_dataset
#args=config,
   #train_dataset=train_dataset,
   #eval_dataset=eval_dataset,
   #peft_config=peft_config,

trainer.train()
trainer.save_model('rewardTrainerExample_n')
#trainer.save_pretrained('rewardTrainerExample_pretrained_n')

#trainer.push_to_hub()
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
print(metrics)

import torch
import pandas as pd
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# tRAIN SMPLE REGRESSION
# Where rejection columns get -1 reward ans accepted +1
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
def train_reward_model(responses, scores):
    print(len(responses))
    encoded_responses = [tokenizer.encode(response, return_tensors='pt',truncation=True, padding='max_length') for response in responses]
    print(len(encoded_responses))
    encoded_responses = torch.cat(encoded_responses, dim=0)

    X_train, X_test, y_train, y_test = train_test_split(encoded_responses.detach().numpy(), scores, test_size=0.2)
    reg_model = LinearRegression().fit(X_train, y_train)

    return reg_model, X_test

# Example data
#responses = ["Response 1", "Response 2", "Response 3"]
#scores = [4, 2, 5]
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train[:1000]")
synth_rude_data_1000_path = '/content/drive/MyDrive/ai_unipi/Deep Learning/synthetic_datasets/wizard_safe_rlhf_dataset_answersConcat_6001_12001.csv'
synth_dataset = pd.read_csv(synth_rude_data_1000_path)
synth_dataset.rename(columns = {'0':'rejected'}, inplace = True)
synth_dataset.rename(columns = {'Unnamed: 0':'id'}, inplace = True)
synth_dataset.drop(['id'], axis=1)
rejected_list = synth_dataset['rejected'].tolist()

responses = dataset['response']
print(responses[0])
responses += rejected_list
print(responses[1000])
scores = []
for i in range(1000):
    scores.append(1)
for i in range(1000):
    scores.append(-1)

reward_model, X_test = train_reward_model(responses, scores)


### predict score
for test_response in X_test[:5]:
    print(tokenizer.decode(test_response))
    print("SCORE:", reward_model.predict([test_response]))


metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
print(metrics)