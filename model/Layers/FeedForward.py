import torch.nn as nn

from model.hyperparameters import processing_device


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension, internal_dimension, dropout_prob):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dimension, internal_dimension, device=processing_device)
        self.linear_2 = nn.Linear(internal_dimension, embedding_dimension, device=processing_device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
