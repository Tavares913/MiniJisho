import torch.nn as nn

from model.Encoder import Encoder

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
