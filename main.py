import pandas as pd
import torch

from model.hyperparameters import batch_size, full_sequence_length, processing_device, num_train_iterations
from utils.utils import encode, decode, get_data_batch

if __name__ == '__main__':
    # report_interval = 1
    # for iteration in range(num_train_iterations):
    #     if (iteration % report_interval == 0 or iteration == num_train_iterations - 1):
    #         print("reporting")
    #
    # x_eng_sents, y_eng_sents, japan_sents = get_data_batch(english_train_data, japanese_train_data)

    print("")