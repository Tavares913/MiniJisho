import torch

batch_size = 32
full_sequence_length = 100
processing_device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_train_iterations = 1
embedding_dimension = 512
dropout_probs = 0.1