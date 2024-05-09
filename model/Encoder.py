import torch.nn as nn

from model.Layers.MultiHeadAttention import MultiHeadAttention
from model.Layers.PositionalEncoding import PositionalEncoding


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension, internal_dimension, dropout_prob):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dimension, internal_dimension)
        self.linear_2 = nn.Linear(internal_dimension, embedding_dimension)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dimension, num_heads, dropout_prob, feedforward_internal_dimension):
        super(EncoderLayer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.multi_head_attention = MultiHeadAttention(embedding_dimension=embedding_dimension, num_heads=num_heads)
        self.dropout_1 = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(dropout_prob)
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = FeedForward(embedding_dimension=embedding_dimension,
                                        internal_dimension=feedforward_internal_dimension, dropout_prob=dropout_prob)

    def forward(self, x, self_attention_mask):
        first_skip_x = x.clone()
        x = self.multi_head_attention(x, self_attention_mask)
        x = self.dropout_1(x)
        x = self.layer_norm_1(x + first_skip_x)
        second_skip_x = x.clone()
        x = self.feed_forward(x)
        x = self.dropout_2(x)
        x = self.layer_norm_2(x + second_skip_x)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, full_sequence_length, num_heads, num_encoder_layers,
                 self_attention_mask, dropout_prob, feedforward_internal_dimension):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size, embedding_dimension)
        self.positional_encoding = PositionalEncoding(embedding_dimension, full_sequence_length)
        self.self_attention_mask = self_attention_mask
        self.embedding_dropout = nn.Dropout(dropout_prob)
        self.encoder_layers = [
            EncoderLayer(embedding_dimension=embedding_dimension, num_heads=num_heads, dropout_prob=dropout_prob,
                         feedforward_internal_dimension=feedforward_internal_dimension) for i
            in range(num_encoder_layers)]

    def forward(self, japan_tokens):
        # token and positional embedding
        x = self.token_embedding(japan_tokens)
        pos_encoding = self.positional_encoding()
        x = self.embedding_dropout(x + pos_encoding)

        # encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, self.self_attention_mask)
        return x
