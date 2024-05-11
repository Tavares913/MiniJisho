import torch.nn as nn
import torch.nn.functional as F

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.hyperparameters import processing_device
from utils.utils import english_characters_to_index, PADDING_TOKEN


class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, embedding_dimension, full_sequence_length, num_heads,
                 num_encoder_layers, num_decoder_layers, dropout_prob, feedforward_internal_dimension):
        super().__init__()
        self.encoder = Encoder(vocab_size=encoder_vocab_size, embedding_dimension=embedding_dimension,
                               full_sequence_length=full_sequence_length, num_heads=num_heads,
                               num_encoder_layers=num_encoder_layers, dropout_prob=dropout_prob,
                               feedforward_internal_dimension=feedforward_internal_dimension)
        self.decoder = Decoder(vocab_size=decoder_vocab_size, embedding_dimension=embedding_dimension,
                               full_sequence_length=full_sequence_length, num_heads=num_heads,
                               dropout_prob=dropout_prob,
                               feedforward_internal_dimension=feedforward_internal_dimension,
                               num_decoder_layers=num_decoder_layers)
        self.last_linear_layer = nn.Linear(embedding_dimension, decoder_vocab_size, device=processing_device)

    def forward(self, encoder_input, decoder_input_x, decoder_input_y, encoder_self_attention_mask,
                decoder_self_attention_mask, decoder_cross_attention_mask):
        x = self.encoder(encoder_input, encoder_self_attention_mask)
        y = self.decoder(encoder_output=x, input=decoder_input_x, self_attention_mask=decoder_self_attention_mask,
                         cross_attention_mask=decoder_cross_attention_mask)
        logits = self.last_linear_layer(y)

        if decoder_input_y is not None:
            batch_size, full_sequence_length, vocab_size = logits.shape
            loss = F.cross_entropy(logits.reshape(batch_size * full_sequence_length, vocab_size),
                                   decoder_input_y.reshape(
                                       batch_size * full_sequence_length), ignore_index=english_characters_to_index[
                    PADDING_TOKEN])  # cross entropy requires shape to be (batch, classes) and (batch) respectively

        else:
            loss = None

        return logits, loss
