import math
import torch

from fast_transformers.builders import RecurrentEncoderBuilder

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RecurrentMusicGenerator(torch.nn.Module):
    def __init__(self, n_tokens, d_model, seq_len,
                 attention_type="full", n_layers=4, n_heads=4,
                 d_query=32, dropout=0.1, softmax_temp=None,
                 attention_dropout=0.1, bits=32, rounds=4,
                 feed_forward_dimensions=2014,
                 chunk_size=32, masked=True):

        super(RecurrentMusicGenerator, self).__init__()

        self.pos_embedding = PositionalEncoding(d_model, max_len=seq_len)
        self.value_embedding = torch.nn.Embedding(n_tokens, d_model)

        self.transformer = RecurrentEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=feed_forward_dimensions,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            softmax_temp=softmax_temp,
            attention_dropout=attention_dropout,
            #bits=bits,
            #rounds=rounds,
            #chunk_size=chunk_size,
            #masked=masked
        ).get()

        self.predictor = torch.nn.Linear(d_model, n_tokens)

    def forward(self, x, i=0, memory=None):
        x = x.view(x.shape[0])
        x = self.value_embedding(x)
        x = self.pos_embedding(x, i)
        y_hat, memory = self.transformer(x, memory)
        y_hat = self.predictor(y_hat)

        return y_hat, memory
