import torch.nn as nn
import torch.nn.functional as F

def calculate_attention_values(q, k, v, mask=None):
    d_k = q.shape[-1]
    weighted_attention_matrix = (q @ k.transpose(-2, -1)) / d_k ** (1/2)
    if mask is not None:
        weighted_attention_matrix = weighted_attention_matrix + mask
    weighted_attention_matrix = F.softmax(weighted_attention_matrix, dim=-1)
    values = weighted_attention_matrix @ v
    return values

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dimension, num_heads):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.head_dimension = embedding_dimension // num_heads
        self.q_layer = nn.Linear(embedding_dimension, embedding_dimension)
        self.k_layer = nn.Linear(embedding_dimension, embedding_dimension)
        self.v_layer = nn.Linear(embedding_dimension, embedding_dimension)
        self.last_linear_layer = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x, mask):
        batch_size, full_sequence_length, embedding_dimension = x.shape

        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)

        # shape q, k, v to be (batch_size, num_heads, full_sequence_length, head_dimension)
        q = q.reshape(batch_size, full_sequence_length, self.num_heads, self.head_dimension).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, full_sequence_length, self.num_heads, self.head_dimension).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, full_sequence_length, self.num_heads, self.head_dimension).permute(0, 2, 1, 3)

        values = calculate_attention_values(q, k, v, mask)
        return values