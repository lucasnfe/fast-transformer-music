#
# Generate MIDI piano pieces with fast transformer.
#
# Author: Lucas N. Ferreira - lucasnfe@gmail.com
#
#

import torch
import math
import argparse

from encoder import decode_midi
from torch.distributions.categorical import Categorical
from fast_transformers.builders import RecurrentEncoderBuilder

START_TOKEN = 388

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(RecurrentGenerator.PositionalEncoding, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, i):
        pos_embedding =  self.pe[0, i:i+1]
        x = torch.cat([x, pos_embedding.expand_as(x)], dim=1)
        return self.dropout(x)

class RecurrentMusicGenerator(torch.nn.Module):
    def __init__(self, n_tokens, d_model, seq_len,
                 attention_type="full", n_layers=4, n_heads=4,
                 d_query=32, dropout=0.1, softmax_temp=None,
                 attention_dropout=0.1, bits=32, rounds=4,
                 feed_forward_dimensions=2014,
                 chunk_size=32, masked=True):

        super(RecurrentMusicGenerator, self).__init__()

        self.pos_embedding = PositionalEncoding(d_model//2, max_len=seq_len)
        self.value_embedding = torch.nn.Embedding(n_tokens, d_model//2)

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

def sample_topk(y_hat, k, filter_value=-float("Inf")):
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = y_hat < torch.topk(y_hat, k)[0][..., -1, None]
    y_hat = y_hat.masked_fill(indices_to_remove, filter_value)

    # Sample from filtered categorical distribution
    random_idx = Categorical(logits=y_hat.view(-1)).sample()
    return random_idx.unsqueeze(0)

def generate(model, prime, n, k=8):
    # Process prime sequence
    memory = None
    y_hat = []
    x_hat = []

    prime_len = len(prime)
    for i in range(prime_len):
        x_hat.append(prime[i])
        y_i, memory = model(x_hat[-1], i=i, memory=memory)
        y_hat.append(y_i)

    # Generate new tokens
    for i in range(prime_len, prime_len + n):
        x_hat.append(sample_topk(y_hat[-1], k))
        yi, memory = model(x_hat[-1], i=i, memory=memory)
        y_hat.append(yi)

    return [int(token) for token in x_hat]

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='generate.py')
    parser.add_argument('--model', type=str, required=True, help="Path to load model from.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--k', type=int, default=10, help="Number k of elements to consider while sampling.")
    parser.add_argument('--vocab_size', type=int, required=True, help="Vocabulary size.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--d_query', type=int, default=32, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    opt = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build linear transformer
    model = RecurrentMusicGenerator(n_tokens=opt.vocab_size,
                                     d_query=opt.d_query,
                                     d_model=opt.d_query * opt.n_heads,
                                     seq_len=opt.seq_len,
                              attention_type="causal-linear",
                                    n_layers=opt.n_layers,
                                     n_heads=opt.n_heads).to(device)

    # Load model
    model.load_state_dict(torch.load(opt.model, map_location=device)["model_state"])

    # Generate piece
    prime = [START_TOKEN]
    prime = torch.tensor(prime).unsqueeze(1)

    piece = generate(model, prime, n=1000, k=opt.k)
    decode_midi(piece, "generated_piece.mid")
    print(piece)
