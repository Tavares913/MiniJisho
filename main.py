import pandas as pd
import torch

from model.hyperparameters import batch_size, context_size, processing_device, max_sequence_length, num_train_iterations
from utils.utils import encode, decode, get_data_batch

if __name__ == '__main__':
    train_data_path= "Datasets/train.csv"
    dev_data_path = "Datasets/dev.csv"
    test_data_path = "Datasets/test.csv"

    # train_translations = pd.read_csv(train_data)
    dev_translations = pd.read_csv(dev_data_path)
    # test_translations = pd.read_csv(test_data)

    english_translations = dev_translations.eng
    japanese_translations = dev_translations.jp

    START_TOKEN = "<START>"
    PADDING_TOKEN = "<PADDING>"
    END_TOKEN = "<END>"


    english_translations = [s for s in english_translations if len(s) < max_sequence_length]
    japanese_translations = [s for s in japanese_translations if len(s) < max_sequence_length]

    english_characters = set("".join(english_translations))
    japanese_characters = set("".join(japanese_translations))
    english_characters.update([START_TOKEN, PADDING_TOKEN, END_TOKEN])
    japanese_characters.update([START_TOKEN, PADDING_TOKEN, END_TOKEN])


    index_to_english_characters = {k:v for k, v in enumerate(english_characters)}
    index_to_japanese_characters = {k:v for k, v in enumerate(japanese_characters)}
    english_characters_to_index = {v:k for k, v in enumerate(english_characters)}
    japanese_characters_to_index = {v:k for k, v in enumerate(japanese_characters)}

    english_train_data = torch.tensor([encode(sent, english_characters_to_index) for sent in english_translations])
    japanese_train_data = torch.tensor(([encode(sent, japanese_characters_to_index)] for sent in japanese_translations))

    report_interval = 1
    for iteration in range(num_train_iterations):
        if (iteration % report_interval == 0 or iteration == num_train_iterations - 1):
            print("reporting")

    x_eng_sents, y_eng_sents, japan_sents = get_data_batch(english_train_data, japanese_train_data)

    print("")