import pandas as pd
import torch

from model.hyperparameters import batch_size, full_sequence_length, processing_device

train_data_path = "datasets/train.csv"
dev_data_path = "datasets/dev.csv"
test_data_path = "datasets/test.csv"

train_translations = pd.read_csv(train_data_path)
dev_translations = pd.read_csv(dev_data_path)
test_translations = pd.read_csv(test_data_path)

english_translations = dev_translations.eng
japanese_translations = dev_translations.jp

START_TOKEN = "<START>"
PADDING_TOKEN = "<PADDING>"
END_TOKEN = "<END>"

filter_values = [
    len(english_translations[i]) < full_sequence_length - 2 and len(japanese_translations[i]) < full_sequence_length - 2
    for
    i
    in range(len(english_translations))]  # -2 because of start and end tokens
english_translations = english_translations[filter_values].reset_index(drop=True)
japanese_translations = japanese_translations[filter_values].reset_index(drop=True)

english_characters = set("".join(english_translations))
japanese_characters = set("".join(japanese_translations))
english_characters.update([START_TOKEN, PADDING_TOKEN, END_TOKEN])
japanese_characters.update([START_TOKEN, PADDING_TOKEN, END_TOKEN])

index_to_english_characters = {k: v for k, v in enumerate(english_characters)}
index_to_japanese_characters = {k: v for k, v in enumerate(japanese_characters)}
english_characters_to_index = {v: k for k, v in enumerate(english_characters)}
japanese_characters_to_index = {v: k for k, v in enumerate(japanese_characters)}


def encode(s, dictionary):
    return [dictionary[c] for c in s]


def decode(ints, dictionary):
    return "".join([dictionary[i] for i in ints])


def tokenize_sentence(sentence, dictionary, add_start_token=True):
    add_to_tokenize = [dictionary[START_TOKEN]]
    padding_tokens_to_add = full_sequence_length - 2 - len(sentence)
    if add_start_token is False:
        add_to_tokenize = []
        padding_tokens_to_add = padding_tokens_to_add + 1
    candidate = add_to_tokenize + encode(sentence, dictionary) + [dictionary[END_TOKEN]] + [dictionary[PADDING_TOKEN]
                                                                                            for i in
                                                                                            range(
                                                                                                padding_tokens_to_add)]
    return torch.tensor(
        add_to_tokenize + encode(sentence, dictionary) + [dictionary[END_TOKEN]] + [dictionary[PADDING_TOKEN]
                                                                                    for i in
                                                                                    range(
                                                                                        padding_tokens_to_add)])


def get_tokenized_data_batch(eng_data, japan_data):
    indices = torch.randint(low=0, high=len(eng_data), size=(batch_size,))
    x_eng_sentences_untokenized = [eng_data[i.item()] for i in indices]
    japan_sentences_untokenized = [japan_data[i.item()] for i in indices]
    x_eng_sentences_tokenized = torch.stack(
        [tokenize_sentence(s, english_characters_to_index) for s in x_eng_sentences_untokenized])
    y_eng_sentences_tokenized = torch.stack(
        [tokenize_sentence(s, english_characters_to_index, add_start_token=False) for s in x_eng_sentences_untokenized])
    japan_sentences_tokenized = torch.stack(
        [tokenize_sentence(s, japanese_characters_to_index) for s in japan_sentences_untokenized])
    return x_eng_sentences_tokenized.to(processing_device), y_eng_sentences_tokenized.to(
        processing_device), japan_sentences_tokenized.to(processing_device)
