import torch.nn as nn

from model.Layers.FeedForward import FeedForward
from model.Layers.MultiHeadAttention import MultiHeadAttention
from model.Layers.PositionalEncoding import PositionalEncoding
from model.hyperparameters import processing_device


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dimension, num_heads, dropout_prob, feedforward_internal_dimension):
        super().__init__()
        self.multi_head_self_attention = MultiHeadAttention(embedding_dimension=embedding_dimension,
                                                            num_heads=num_heads)
        self.multi_head_cross_attention = MultiHeadAttention(embedding_dimension=embedding_dimension,
                                                             num_heads=num_heads)
        self.dropout_1 = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(dropout_prob)
        self.dropout_3 = nn.Dropout(dropout_prob)
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension, device=processing_device)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension, device=processing_device)
        self.layer_norm_3 = nn.LayerNorm(embedding_dimension, device=processing_device)
        self.feed_forward = FeedForward(embedding_dimension=embedding_dimension,
                                        internal_dimension=feedforward_internal_dimension, dropout_prob=dropout_prob)

    def forward(self, encoder_output, input, self_attention_mask, cross_attention_mask):
        first_skip_y = input.clone()
        y = self.multi_head_self_attention(input, mask=self_attention_mask)
        y = self.dropout_1(y)
        y = self.layer_norm_1(y + first_skip_y)
        second_skip_y = y.clone()
        y = self.multi_head_cross_attention(cur_encoding=y, mask=cross_attention_mask, cross_encoding=encoder_output)
        y = self.dropout_2(y)
        y = self.layer_norm_2(y + second_skip_y)
        third_skip_y = y.clone()
        y = self.feed_forward(y)
        y = self.dropout_3(y)
        y = self.layer_norm_3(y + third_skip_y)
        return y


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dimension, full_sequence_length, dropout_prob, num_heads,
                 feedforward_internal_dimension, num_decoder_layers, self_attention_mask, cross_attention_mask):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.full_sequence_length = full_sequence_length
        self.self_attention_mask = self_attention_mask
        self.cross_attention_mask = cross_attention_mask
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension, device=processing_device)
        self.embedding_dropout = nn.Dropout(dropout_prob)
        self.positional_encoding = PositionalEncoding(embedding_dimension, full_sequence_length)
        self.decoder_layers = [
            DecoderLayer(embedding_dimension=embedding_dimension, num_heads=num_heads, dropout_prob=dropout_prob,
                         feedforward_internal_dimension=feedforward_internal_dimension) for i in
            range(num_decoder_layers)]

    def forward(self, encoder_output, input):
        # token and positional embedding
        y = self.token_embedding(input)
        positional_encodings = self.positional_encoding()
        y = self.embedding_dropout(y + positional_encodings)

        # decoder layers
        for decoder_layer in self.decoder_layers:
            y = decoder_layer(encoder_output, y, None, None)
        return y
