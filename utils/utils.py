import torch

from model.hyperparameters import context_size, batch_size

def encode(s, dict):
    return [dict[c] for c in s]

def decode(ints, dict):
    return "".join([dict[i] for i in ints])

def get_data_batch(eng_data, japan_data):
    indexes = torch.randint(low=0, high=len(data), size=(batch_size,))
    x_eng_sents = torch.stack([eng_data[i] for i in indexes])
    y_eng_sents = torch.stack([[0] + eng_data[i][1:] for i in indexes], dim=0)
    japan_sents = torch.stack([japan_data[i] for i in indexes])
    return x_eng_sents, y_eng_sents, japan_sents