import torch.nn.functional as F
import torch.optim

from model.Transformer import Transformer
from model.hyperparameters import full_sequence_length, num_train_iterations, \
    embedding_dimension, num_encoder_layers, num_multi_head_attention_heads, dropout_prob, \
    feedforward_internal_dimension, num_decoder_layers, learning_rate, processing_device
from utils.utils import english_translations_train, english_translations_dev, japanese_translations_train, \
    japanese_translations_dev, \
    get_tokenized_data_batch, \
    english_characters, english_characters_to_index, index_to_english_characters, japanese_characters, \
    japanese_characters_to_index, create_masks, \
    tokenize_sentence, END_TOKEN, PADDING_TOKEN, START_TOKEN


def forward_batch(in_eval_mode=False):
    if in_eval_mode is False:
        english_translations = english_translations_train
        japanese_translations = japanese_translations_train
        model.train()
    else:
        english_translations = english_translations_dev
        japanese_translations = japanese_translations_dev
        model.eval()

    x_eng_sentences_tokenized, y_eng_sentences_tokenized, japan_sentences_tokenized, x_eng_sentences_untokenized, japan_sentences_untokenized = get_tokenized_data_batch(
        english_translations, japanese_translations)

    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
        x_eng_sentences_untokenized, japan_sentences_untokenized)

    logits, loss = model(encoder_input=japan_sentences_tokenized, decoder_input_x=x_eng_sentences_tokenized,
                         decoder_input_y=y_eng_sentences_tokenized,
                         encoder_self_attention_mask=encoder_self_attention_mask,
                         decoder_self_attention_mask=decoder_self_attention_mask,
                         decoder_cross_attention_mask=decoder_cross_attention_mask)

    return logits, loss


def translate_japanese_to_english(japan_sentence):
    english_sentence_untokenized = ""
    japan_sentence_untokenized = japan_sentence
    for char_idx in range(full_sequence_length):
        x_eng_sentence_tokenized, japan_sentence_tokenized = tokenize_sentence(
            english_sentence_untokenized, dictionary=english_characters_to_index,
            add_end_token=False), tokenize_sentence(
            japan_sentence_untokenized, dictionary=japanese_characters_to_index)

        x_eng_sentence_tokenized_input = torch.stack([x_eng_sentence_tokenized]).to(processing_device)
        japan_sentence_tokenized_input = torch.stack([japan_sentence_tokenized]).to(processing_device)

        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            [english_sentence_untokenized], [japan_sentence_untokenized])

        model.eval()
        logits, _ = model(encoder_input=japan_sentence_tokenized_input, decoder_input_x=x_eng_sentence_tokenized_input,
                          decoder_input_y=None,
                          encoder_self_attention_mask=encoder_self_attention_mask,
                          decoder_self_attention_mask=decoder_self_attention_mask,
                          decoder_cross_attention_mask=decoder_cross_attention_mask)
        next_token_probability_distribution_raw = logits[0][
            char_idx]  # get the first (and only) batch + current timestep
        next_token_probability_distribution_softmax = F.softmax(next_token_probability_distribution_raw, dim=-1)
        next_token_index = torch.multinomial(next_token_probability_distribution_softmax, num_samples=1)
        next_token = index_to_english_characters[next_token_index.item()]
        english_sentence_untokenized = english_sentence_untokenized + next_token
        if (next_token == END_TOKEN or next_token == START_TOKEN or next_token == PADDING_TOKEN):
            break
    return english_sentence_untokenized


if __name__ == '__main__':
    # train model
    reporting_interval = 100
    loss_history = []

    m = Transformer(encoder_vocab_size=len(japanese_characters), decoder_vocab_size=len(english_characters),
                    embedding_dimension=embedding_dimension, full_sequence_length=full_sequence_length,
                    num_heads=num_multi_head_attention_heads, num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dropout_prob=dropout_prob, feedforward_internal_dimension=feedforward_internal_dimension)

    model = m.to(processing_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for iteration in range(num_train_iterations):
        _, train_loss = forward_batch()
        loss_history.append(train_loss)

        if iteration % reporting_interval == 0 or iteration == num_train_iterations - 1:
            _, val_loss = forward_batch(in_eval_mode=True)
            print(f"Loss at iteration {iteration} - Train loss: {train_loss}; Val loss: {val_loss}")
            print(f"Translation 1: {translate_japanese_to_english('今、学校に行きます!')}")  # I'm going to school now!
            print(f"Translation 2: {translate_japanese_to_english('仕事はめっちゃ難しい')}")  # My job is really hard

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

    # evaluate model
    model.eval()
