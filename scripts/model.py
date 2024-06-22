from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import numpy as np
import evaluate
import torch
from tokenizer import *


## Use a DataCollator to get batches during training
from transformers import DefaultDataCollator, DataCollatorForLanguageModeling

#data_collator = DefaultDataCollator()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)





'''
# call Language model
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device) ; # train on GPU if possible
#model.gradient_checkpointing_enable()
'''


# Check for MPS (Mac GPU support) or CUDA, fallback to CPU if none are available
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Load the model and move it to the selected device
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.to(device)

# Uncomment this line if you want to enable gradient checkpointing
# model.gradient_checkpointing_enable()

print(f'Model is using device: {device}')



def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir='./results', #this will be usd in the pipeline for inference
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True,  # Enable mixed precision training
    logging_steps=10,
    save_steps=1000,  # Save a checkpoint every 1000 steps
    save_total_limit=5,  # Keep only the last 2 checkpoints
    save_strategy="steps",
)
#eval_strategy="epoch",  # Use eval_strategy instead of evaluation_strategy
#eval_steps=10,
#do_eval=False,
#eval_accumulation_steps=1,

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset,
)
#compute_metrics=compute_metrics

# get pretraining metric evaluation
#print("BEFORE TRAINING, trying evaluation metrics:")
#pre_eval_metrics = trainer.evaluate(eval_dataset)
#print(pre_eval_metrics)
#########
trainer.train()
print("AFTER TRAINING, trying evaluation metrics:")
pre_eval_metrics = trainer.evaluate(eval_dataset)
print(pre_eval_metrics)

torch.cuda.empty_cache()
evaluation_results = trainer.evaluate()

### save model and tokenizer locally
trainer.model.save_pretrained("model_trained")
tokenizer.save_pretrained("tokenizer_trained")


### save model and use it later for inference, evaluation and deployment
saved_model_path = 'model/model_trained'
saved_tokenizer_path = 'model/tokenizer_trained'