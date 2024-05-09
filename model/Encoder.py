import torch.nn as nn

from model.hyperparameters import dropout_probs
from model.Layers.PositionalEncoding import PositionalEncoding
from model.Layers.MultiHeadAttention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dimension, num_heads):
        super(EncoderLayer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dimension=embedding_dimension, num_heads=num_heads)

    def forward(self, x, self_attention_mask):
        first_skip_x = x.clone()
        x = self.multi_head_attention(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, full_sequence_length, num_heads, num_encoder_layers, self_attention_mask):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size, embedding_dimension)
        self.positional_encoding = PositionalEncoding(embedding_dimension, full_sequence_length)
        self.self_attention_mask = self_attention_mask
        self.embedding_dropout = nn.Dropout(dropout_probs)
        self.encoder_layers = [EncoderLayer(embedding_dimension=embedding_dimension, num_heads=num_heads) for i in range(num_encoder_layers)]

    def forward(self, japan_tokens):
        # token and positional embedding
        x = self.token_embedding(japan_tokens)
        pos_encoding = self.positional_encoding()
        x = self.embedding_dropout(x + pos_encoding)

        # encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, self.self_attention_mask)

        return x
