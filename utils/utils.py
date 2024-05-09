import pandas as pd
import torch

from model.hyperparameters import batch_size, full_sequence_length

train_data_path = "datasets/train.csv"
dev_data_path = "datasets/dev.csv"
test_data_path = "datasets/test.csv"

# train_translations = pd.read_csv(train_data)
dev_translations = pd.read_csv(dev_data_path)
# test_translations = pd.read_csv(test_data)

english_translations = dev_translations.eng
japanese_translations = dev_translations.jp

START_TOKEN = "<START>"
PADDING_TOKEN = "<PADDING>"
END_TOKEN = "<END>"

filter_values = [
    len(english_translations[i]) < full_sequence_length and len(japanese_translations[i]) < full_sequence_length for i
    in range(len(english_translations))]
english_translations = english_translations[filter_values]
japanese_translations = japanese_translations[filter_values]

english_characters = set("".join(english_translations))
japanese_characters = set("".join(japanese_translations))
english_characters.update([START_TOKEN, PADDING_TOKEN, END_TOKEN])
japanese_characters.update([START_TOKEN, PADDING_TOKEN, END_TOKEN])

index_to_english_characters = {k: v for k, v in enumerate(english_characters)}
index_to_japanese_characters = {k: v for k, v in enumerate(japanese_characters)}
english_characters_to_index = {v: k for k, v in enumerate(english_characters)}
japanese_characters_to_index = {v: k for k, v in enumerate(japanese_characters)}

def encode(s, dict):
    return [dict[c] for c in s]

def decode(ints, dict):
    return "".join([dict[i] for i in ints])

def tokenize_sentence(sentence, dict):
    return torch.tensor([dict[START_TOKEN]] + encode(sentence, dict) + [dict[END_TOKEN]] + [dict[PADDING_TOKEN] for i in range(full_sequence_length - 2 - len(sentence))])
def get_tokenized_data_batch(eng_data, japan_data):
    indices = torch.randint(low=0, high=len(eng_data), size=(batch_size,))
    x_eng_sents_untokenized = [eng_data[i.item()] for i in indices]
    y_eng_sents_untokenized = [eng_data[i.item()][1:] for i in indices]
    japan_sents_untokenized = [japan_data[i.item()] for i in indices]
    x_eng_sents_tokenized = torch.stack([tokenize_sentence(s, english_characters_to_index) for s in x_eng_sents_untokenized])
    y_eng_sents_tokenized = torch.stack([tokenize_sentence(s, english_characters_to_index) for s in y_eng_sents_untokenized])
    japan_sents_tokenized = torch.stack([tokenize_sentence(s, japanese_characters_to_index) for s in japan_sents_untokenized])
    return x_eng_sents_tokenized, y_eng_sents_tokenized, japan_sents_tokenized
