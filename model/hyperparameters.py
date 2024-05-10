import torch

learning_rate = 1e-4
batch_size = 32
full_sequence_length = 100
processing_device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_train_iterations = 5000
embedding_dimension = 512
num_multi_head_attention_heads = 8
num_encoder_layers = 5
num_decoder_layers = 5
dropout_prob = 0.1
feedforward_internal_dimension = 4 * embedding_dimension
