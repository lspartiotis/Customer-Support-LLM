from reward_model import *


# 0. imports
import torch
from transformers import GPT2Tokenizer,pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")


# 1. load a pretrained model
model = AutoModelForCausalLMWithValueHead.from_pretrained("/content/drive/MyDrive/ai_unipi/Deep Learning/model_trained") ### finetuned
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
#dataset = dataset.rename_column("instruction", "query")
#dataset = dataset.remove_columns(["meta", "completion"])
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

# 3. encode a query
query_txt = f"Customer: Hello I would like to cancel my order {tokenizer.eos_token}"
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 100,
}
#### Get response from SFTModel
response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
response_txt = tokenizer.decode(response_tensor[0])
print(response_txt)

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
#### Compute reward score
#print(query_tensor, response_tensor, response_txt, tokenizer.decode(response_tensor[0]))
#texts = [q + r for q, r in zip(query_tensor, response_tensor)]
#pipe_outputs = reward_model(texts)
#rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

print("RESPONSE:", response_txt)
print("TRAIN STATS: \n", train_stats)
#ppo_trainer.log_stats(train_stats, batch, rewards)



# above but wwith multiple data ---BAD RESULTS AFTER A WHILE
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch
from transformers import pipeline
reward_model = pipeline("text-classification", model="rewardTrainerExample")
# get models
model = AutoModelForCausalLMWithValueHead.from_pretrained("model_trained").to(device) ### finetuned
model_ref = create_reference_model(model).to(device)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# initialize trainer
ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)
# create a ppo trainer
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer, dataset)

i = 0
for query_txt in ppo_trainer.dataset['instruction']:
    # encode a query
    query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(device)

    # get model response
    response_tensor = ppo_trainer.generate(query_tensor[0], max_length=128, num_return_sequences=1).to(device) #, return_prompt=False

    # define a reward for response
    # (this could be any reward such as human feedback or output from another model)
    decoded_text = tokenizer.decode(response_tensor[0], skip_special_tokens=True)
    pipe_outputs = reward_model.predict([decoded_text])

    rewards = [output for output in pipe_outputs]
    reward = reward_model(decoded_text)
    reward = reward[0]['score']
    score_tensor = torch.tensor([reward]).to(device)
    score_list = [score_tensor]

    # train model for one step with ppo
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], score_list)
    temp_dict = {
        'query':query_txt,
        'response': tokenizer.batch_decode(response_tensor, skip_special_tokens=True),
        'input_ids':query_tensor,
        'score_list': score_list
    }
    i += 1


    