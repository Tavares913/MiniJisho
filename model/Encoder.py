import torch
import torch.nn as nn

from model.hyperparameters import dropout_probs

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dimension, full_sequence_length):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.full_sequence_length = full_sequence_length

    def forward(self):
        even_indices = torch.arange(0, self.embedding_dimension, 2).float()
        denominator = torch.pow(10000, even_indices / self.embedding_dimension) # both the sin and cos versions end up using the same denominator
        all_positions = torch.arange(self.full_sequence_length).reshape(-1, 1)
        even_position_encodings = torch.sin(all_positions / denominator)
        odd_position_encodings = torch.cos(all_positions / denominator)
        combined_position_encodings = torch.flatten(torch.stack([even_position_encodings, odd_position_encodings], dim=2), start_dim=1, end_dim=2) # stack on dim 2 then flatten to read off dim 1, then 2, then 1 etc.
        return combined_position_encodings


class Encoder(nn.Module):
    def __init__(self, lang_to_idx_dict, embedding_dimension, full_sequence_length):
        super().__init__()
        self.vocab_size = len(lang_to_idx_dict)
        self.token_embedding = nn.Embedding(self.vocab_size, embedding_dimension)
        self.positional_encoding = PositionalEncoding(embedding_dimension, full_sequence_length)
        self.embedding_dropout = nn.Dropout(dropout_probs)

    def forward(self, japan_tokens):
        x = self.token_embedding(japan_tokens)
        pos_encoding = self.positional_encoding()
        x = self.embedding_dropout(x + pos_encoding)
        return x
