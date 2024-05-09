from model.hyperparameters import batch_size, full_sequence_length, processing_device, num_train_iterations, embedding_dimension, num_encoder_layers, num_multi_head_attention_heads
from utils.utils import english_translations, japanese_translations, get_tokenized_data_batch, english_characters, japanese_characters
from model.Transformer import Transformer

if __name__ == '__main__':
    model = Transformer(encoder_vocab_size=len(japanese_characters), decoder_vocab_size=len(english_characters), embedding_dimension=embedding_dimension, full_sequence_length=full_sequence_length, num_heads=num_multi_head_attention_heads, num_encoder_layers=num_encoder_layers)
    for iteration in range(num_train_iterations):
        x_eng_sents_tokenized, y_eng_sents_tokenized, japan_sents_tokenized = get_tokenized_data_batch(english_translations, japanese_translations)
        res = model(encoder_input=japan_sents_tokenized, decoder_input_x=x_eng_sents_tokenized, decoder_input_y=y_eng_sents_tokenized)
        print(res)

    print("")