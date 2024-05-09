import torch

batch_size = 4
full_sequence_length = 100
processing_device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_train_iterations = 1
embedding_dimension = 512
num_multi_head_attention_heads = 8
num_encoder_layers = 1
dropout_prob = 0.1
feedforward_internal_dimension = 4 * embedding_dimension
