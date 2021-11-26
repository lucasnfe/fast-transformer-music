#
# Generate MIDI piano pieces with fast transformer.
#
# Author: Lucas N. Ferreira - lucasnfe@gmail.com
#
#

import torch
import argparse

from encoder import decode_midi
from model_generate import RecurrentMusicGenerator
from torch.distributions.categorical import Categorical

START_TOKEN = 388

def sample_topk(y_hat, k):
    values, indices = torch.topk(y_hat.view(-1), k)
    random_idx = Categorical(logits=values).sample()
    return indices[random_idx].unsqueeze(0)

def generate(model, prime, n, k=10):
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
    parser.add_argument('--vocab_size', type=int, required=True, help="Vocabulary size.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--d_query', type=int, default=32, help="Dimension of the query matrix.")
    opt = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build linear transformer
    model = RecurrentMusicGenerator(n_tokens=opt.vocab_size,
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

    piece = generate(model, prime, n=1000)
    decode_midi(piece, "generated_piece.mid")
    print(piece)
