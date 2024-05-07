import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Hyperparameters
    max_sequence_length = 100

    train_data = "Datasets/train.csv"
    dev_data = "Datasets/dev.csv"
    test_data = "Datasets/test.csv"

    # train_translations = pd.read_csv(train_data)
    dev_translations = pd.read_csv(dev_data)
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

    print("")