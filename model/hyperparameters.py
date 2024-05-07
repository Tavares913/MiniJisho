import torch

batch_size = 32
context_size = 32
processing_device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_sequence_length = 100
num_train_iterations = 1