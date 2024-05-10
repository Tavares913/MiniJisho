import torch.optim

from model.Transformer import Transformer
from model.hyperparameters import full_sequence_length, num_train_iterations, \
    embedding_dimension, num_encoder_layers, num_multi_head_attention_heads, dropout_prob, \
    feedforward_internal_dimension, num_decoder_layers, learning_rate, processing_device
from utils.utils import english_translations, japanese_translations, get_tokenized_data_batch, english_characters, \
    japanese_characters

if __name__ == '__main__':
    reporting_interval = 10
    loss_history = []

    m = Transformer(encoder_vocab_size=len(japanese_characters), decoder_vocab_size=len(english_characters),
                    embedding_dimension=embedding_dimension, full_sequence_length=full_sequence_length,
                    num_heads=num_multi_head_attention_heads, num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dropout_prob=dropout_prob, feedforward_internal_dimension=feedforward_internal_dimension)
    model = m.to(processing_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iteration in range(num_train_iterations):
        x_eng_sentences_tokenized, y_eng_sentences_tokenized, japan_sentences_tokenized = get_tokenized_data_batch(
            english_translations, japanese_translations)

        logits, loss = model(encoder_input=japan_sentences_tokenized, decoder_input_x=x_eng_sentences_tokenized,
                             decoder_input_y=y_eng_sentences_tokenized)

        loss_history.append(loss)

        if iteration % reporting_interval == 0 or iteration == num_train_iterations - 1:
            print(f"Loss at iteration {iteration}: {loss}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"Iteration {iteration}")

    print("")
