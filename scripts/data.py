from datasets import load_dataset
import pandas as pd
import re
import string
from spellchecker import SpellChecker


### Load the dataset from huggingface
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
sample_on = False


dataset_choice = 0



base_dataset: pd.DataFrame = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train").to_pandas()
base_dataset = base_dataset[["instruction", "category", "intent", "response"]]
print(base_dataset.describe()) # -> Num rows and number of unique values per column
print(base_dataset['intent'].value_counts())
print(base_dataset['category'].value_counts()) # -> Pretty heavily concentrated on ACCOUNT, ORDER and REFUND categories.


df = dataset['train'].to_pandas()
df_grouped = df.groupby(['intent', 'category']).apply(lambda x: x)
print(df['category'].value_counts())
df.groupby(['category', 'intent']).size() # count different intent categories in each category

### clean the data
### creadit to: https://www.kaggle.com/code/abdulrahmanatef/start-with-text-preprocessing
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', string.punctuation))
# TODO: keep emoticons and emojis, since they convey emotion
# and can give feedback on customer experience
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalize_elongated_words(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def correct_spellings(text):
    spell = SpellChecker()
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            # unrecognized word
            if spell.correction(word) is None:
                corrected_text.append(word)
            else:
                corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    #print(text, corrected_text, '\n')
    return " ".join(corrected_text)


def clean_text(text):
    # remove mentions and hashtags
    text = re.sub(r'@\w+\s*', '', text)
    text = re.sub(r'#\w+\s*', '', text)
    text = remove_emoji(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # add space after punctuations
    text = re.sub(r'([..])(\S)', r'\1 \2', text)
    text = re.sub(r'([!?-])(\S)', r'\1 \2', text)
    text = remove_punctuation(text)
    # Lowercase the text
    text = text.lower()
    text = normalize_elongated_words(text)
    # Fix spelling
    #text = correct_spellings(text)

    return text

def clean_text_bitext_x(text):
    # remove mentions and hashtags

    text['cleaned_text_x'] = re.sub(r'@\w+\s*', '', text['instruction'])
    text['cleaned_text_x'] = re.sub(r'#\w+\s*', '', text['cleaned_text_x'])
    text['cleaned_text_x'] = remove_emoji(text['cleaned_text_x'])
    # Remove URLs
    text['cleaned_text_x'] = re.sub(r'http\S+|www\S+|https\S+', '', text['cleaned_text_x'], flags=re.MULTILINE)
    # add space after punctuations
    text['cleaned_text_x'] = re.sub(r'([..])(\S)', r'\1 \2', text['cleaned_text_x'])
    text['cleaned_text_x'] = re.sub(r'([!?-])(\S)', r'\1 \2', text['cleaned_text_x'])
    text['cleaned_text_x'] = remove_punctuation(text['cleaned_text_x'])
    # Lowercase the text
    text['cleaned_text_x'] = text['cleaned_text_x'].lower()
    text['cleaned_text_x'] = normalize_elongated_words(text['cleaned_text_x'])
    # Fix spelling
    #text = correct_spellings(text)

    return text


def clean_text_bitext_y(text):
    # remove mentions and hashtags
    text['cleaned_text_y'] = re.sub(r'@\w+\s*', '', text['response'])
    text['cleaned_text_y'] = re.sub(r'#\w+\s*', '', text['cleaned_text_y'])
    text['cleaned_text_y'] = remove_emoji(text['cleaned_text_y'])
    # Remove URLs
    text['cleaned_text_y'] = re.sub(r'http\S+|www\S+|https\S+', '', text['cleaned_text_y'], flags=re.MULTILINE)
    # add space after punctuations
    text['cleaned_text_y'] = re.sub(r'([..])(\S)', r'\1 \2', text['cleaned_text_y'])
    text['cleaned_text_y'] = re.sub(r'([!?-])(\S)', r'\1 \2', text['cleaned_text_y'])
    text['cleaned_text_y'] = remove_punctuation(text['cleaned_text_y'])
    # Lowercase the text
    text['cleaned_text_y'] = text['cleaned_text_y'].lower()
    text['cleaned_text_y'] = normalize_elongated_words(text['cleaned_text_y'])
    # Fix spelling
    #text = correct_spellings(text)

    return text



if sample_on == True:
    dataset = dataset.map(clean_text_bitext_x)
    dataset = dataset.map(clean_text_bitext_y)
else:
    dataset['train'] = dataset['train'].map(clean_text_bitext_x)
    dataset['train'] = dataset['train'].map(clean_text_bitext_y)
dataset_pairs = dataset
print(dataset_pairs)

###### TRAIN WITHOUT SPELL CHECKING CAUSE IT IS TIME CONSUMING ######



from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def combine_query_response(example):
    example['dialogue_pair'] = example['cleaned_text_x'] + " <|endoftext|> " + example['cleaned_text_y']
    return example

def combine_query_response_tokened(example):
    example['dialogue_pair'] = "\nCustomer: " + example['cleaned_text_x'] + tokenizer.eos_token + "\nAssistant: " + example['cleaned_text_y']
    return example

dataset['train'] = dataset['train'].map(combine_query_response_tokened) #combine_query_response

import pandas as pd
df = pd.DataFrame(dataset['train'])
print(df.head())


