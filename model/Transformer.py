import torch.nn as nn

from model.Encoder import Encoder
from utils.utils import english_characters

class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embedding_dimension, full_sequence_length, num_heads, num_encoder_layers):
        super().__init__()
        self.encoder = Encoder(vocab_size=encoder_vocab_size, embedding_dimension=embedding_dimension, full_sequence_length=full_sequence_length, num_heads=num_heads, self_attention_mask=None, num_encoder_layers=num_encoder_layers)

    def forward(self, encoder_input, decoder_input_x, decoder_input_y, y=None):
        x = self.encoder(encoder_input)
        return x

