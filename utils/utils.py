import pandas as pd
import torch

from model.hyperparameters import batch_size, full_sequence_length, processing_device

NEG_INFTY = -1e9

train_data_path = "datasets/train.csv"
dev_data_path = "datasets/dev.csv"
test_data_path = "datasets/test.csv"

train_translations = pd.read_csv(train_data_path)
dev_translations = pd.read_csv(dev_data_path)
test_translations = pd.read_csv(test_data_path)

english_translations_train = train_translations.eng
english_translations_dev = dev_translations.eng
english_translations_test = test_translations.eng
japanese_translations_train = train_translations.jp
japanese_translations_dev = dev_translations.jp
japanese_translations_test = test_translations.jp

START_TOKEN = "<START>"
PADDING_TOKEN = "<PADDING>"
END_TOKEN = "<END>"

filter_values_train = [
    len(english_translations_train[i]) < full_sequence_length - 2 and len(
        japanese_translations_train[i]) < full_sequence_length - 2
    for
    i
    in range(len(english_translations_train))]  # -2 because of start and end tokens
filter_values_dev = [
    len(english_translations_dev[i]) < full_sequence_length - 2 and len(
        japanese_translations_dev[i]) < full_sequence_length - 2
    for
    i
    in range(len(english_translations_dev))]  # -2 because of start and end tokens
filter_values_test = [
    len(english_translations_test[i]) < full_sequence_length - 2 and len(
        japanese_translations_test[i]) < full_sequence_length - 2
    for
    i
    in range(len(english_translations_test))]  # -2 because of start and end tokens
english_translations_train = english_translations_train[filter_values_train].reset_index(drop=True)
english_translations_dev = english_translations_dev[filter_values_dev].reset_index(drop=True)
english_translations_test = english_translations_test[filter_values_test].reset_index(drop=True)
japanese_translations_train = japanese_translations_train[filter_values_train].reset_index(drop=True)
japanese_translations_dev = japanese_translations_dev[filter_values_dev].reset_index(drop=True)
japanese_translations_test = japanese_translations_test[filter_values_test].reset_index(drop=True)

english_characters = set("".join(english_translations_train))
japanese_characters = set("".join(japanese_translations_train))
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


def tokenize_sentence(sentence, dictionary, add_start_token=True, add_end_token=True):
    add_start_to_tokenize = [dictionary[START_TOKEN]]
    add_end_to_tokenize = [dictionary[END_TOKEN]]
    padding_tokens_to_add = full_sequence_length - 2 - len(sentence)
    if add_start_token is False:
        add_start_to_tokenize = []
        padding_tokens_to_add = padding_tokens_to_add + 1
    if add_end_token is False:
        add_end_to_tokenize = []
        padding_tokens_to_add = padding_tokens_to_add + 1

    return torch.tensor(
        add_start_to_tokenize + encode(sentence, dictionary) + add_end_to_tokenize + [dictionary[PADDING_TOKEN]
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
        processing_device), japan_sentences_tokenized.to(
        processing_device), x_eng_sentences_untokenized, japan_sentences_untokenized


def create_masks(eng_batch_untokenized_sentences, jap_batch_untokenized_sentences):
    num_in_batch = len(eng_batch_untokenized_sentences)
    single_sentence_look_ahead_mask = torch.triu(torch.full([full_sequence_length, full_sequence_length], True),
                                                 diagonal=1)  # upper triangular filled with True with diagonal entries = 0

    # bases with the padding tokens masked
    encoder_padding_mask = torch.full([num_in_batch, full_sequence_length, full_sequence_length],
                                      False)  # (batch, jap, jap)
    decoder_padding_mask_self_attention = torch.full([num_in_batch, full_sequence_length, full_sequence_length],
                                                     False)  # (batch, eng, eng)
    decoder_padding_mask_cross_attention = torch.full([num_in_batch, full_sequence_length, full_sequence_length],
                                                      False)  # (batch, eng, jap)

    for idx in range(num_in_batch):
        eng_sentence_length, jap_sentence_length = len(eng_batch_untokenized_sentences[idx]), len(
            jap_batch_untokenized_sentences[idx])

        indices_of_padding_in_eng_sentence = torch.arange(eng_sentence_length + 1, full_sequence_length)
        indices_of_padding_in_jap_sentence = torch.arange(jap_sentence_length + 1, full_sequence_length)

        encoder_padding_mask[idx, :, indices_of_padding_in_jap_sentence] = True
        encoder_padding_mask[idx, indices_of_padding_in_jap_sentence,
        :] = True  # mask all of the padding in the japanese sentence
        decoder_padding_mask_self_attention[idx, :, indices_of_padding_in_eng_sentence] = True
        decoder_padding_mask_self_attention[idx, indices_of_padding_in_eng_sentence,
        :] = True  # mask all of the padding in the english sentence
        decoder_padding_mask_cross_attention[idx, :, indices_of_padding_in_eng_sentence] = True
        decoder_padding_mask_cross_attention[idx, indices_of_padding_in_jap_sentence,
        :] = True  # mask all of the padding in both the eng and japanese sentences

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY,
                                              0)  # Where there's padding, put NEG_INFINITY, else 0
    decoder_self_attention_mask = torch.where(single_sentence_look_ahead_mask + decoder_padding_mask_self_attention,
                                              NEG_INFTY,
                                              0)  # where is should be masked either because its a lookahead or its padding, put NEG_INFINITY, else 0
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY,
                                               0)  # Where theres padding, put NEG_INFINITY, else 0
    return encoder_self_attention_mask.to(processing_device), decoder_self_attention_mask.to(
        processing_device), decoder_cross_attention_mask.to(processing_device)
