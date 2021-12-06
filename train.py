import os
import time
import math
import torch
import argparse

from vgmidi import VGMidi
from radam import RAdam

from dmll import discretized_mix_logistic_loss
from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos_embedding =  self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x = torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)

class MusicGenerator(torch.nn.Module):
    def __init__(self, n_tokens, d_model, seq_len, mixtures,
                 attention_type="full", n_layers=4, n_heads=4,
                 d_query=32, dropout=0.1, softmax_temp=None,
                 attention_dropout=0.1, bits=32, rounds=4,
                 feed_forward_dimensions=1024,
                 chunk_size=32, masked=True):

        super(MusicGenerator, self).__init__()

        self.pos_embedding = PositionalEncoding(d_model//2, max_len=seq_len)
        self.value_embedding = torch.nn.Embedding(n_tokens, d_model//2)

        self.transformer = TransformerEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=feed_forward_dimensions,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            softmax_temp=softmax_temp,
            attention_dropout=attention_dropout,
            bits=bits,
            rounds=rounds,
            chunk_size=chunk_size,
            masked=masked
        ).get()

        hidden_size = n_heads * d_query
        self.predictor = torch.nn.Linear(hidden_size, mixtures * 3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.value_embedding(x)
        x = self.pos_embedding(x)

        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        y_hat = self.transformer(x, attn_mask=triangular_mask)
        y_hat = self.predictor(y_hat)

        return y_hat

def save_model(model, optimizer, epoch, save_to):
    model_path = save_to.format(epoch)
    torch.save(
        dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch
        ),
        model_path)

def loss(y, y_hat):
    log2 = 0.6931471805599453
    y_hat = y_hat.permute(0, 2, 1).contiguous()
    N, C, L = y_hat.shape
    l = discretized_mix_logistic_loss(y_hat, y.view(N, L, 1))
    bpd = l.item() / log2
    return l, bpd

def train(model, train_data, test_data, epochs, lr, save_to):
    best_model = None
    best_val_loss = float('inf')

    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Train model for one epoch
        train_step(model, train_data, epoch, lr, optimizer)

        # Evaluate model on test set
        val_loss = evaluate(model, test_data)

        elapsed = time.time() - epoch_start_time

        # Compute validation perplexity
        val_ppl = math.exp(val_loss)

        # Log training statistics for this epoch
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')

        # Save best model so far
        if val_loss < best_val_loss:
            print(f'Validation loss improved from {best_val_loss:5.2f} to {val_loss:5.2f}.'
                  f'Saving model to {save_to}.')

            best_val_loss = val_loss
            save_model(model, optimizer, epoch, save_to)

        print('-' * 89)

    return best_model

def train_step(model, train_data, epoch, lr, optimizer, log_interval=100):
    model.train()
    start_time = time.time()

    total_loss = 0
    for batch, (x, y) in enumerate(train_data):
        # Forward pass
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)

        # Backward pass
        optimizer.zero_grad()
        l, bpd = loss(y, y_hat)
        l.backward()
        optimizer.step()

        # Log training statistics
        total_loss += l.item()
        if batch % log_interval == 0 and batch > 0:
            log_stats(optimizer, epoch, batch, len(train_data), total_loss, start_time, log_interval)
            total_loss = 0
            start_time = time.time()

def log_stats(optimizer, epoch, batch, num_batches, total_loss, start_time, log_interval):
    # Get current learning rate
    lr = optimizer.get_lr()[0]

    # Compute duration of each batch
    ms_per_batch = (time.time() - start_time) * 1000 / log_interval

    # Compute current loss
    cur_loss = total_loss / log_interval

    # compute current perplexity
    ppl = math.exp(cur_loss)

    print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
          f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
          f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')

def evaluate(model, test_data):
    model.eval()

    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_data):
            x = x.to(device)
            y = y.to(device)

            # Evaluate
            y_hat = model(x)
            l, bpd = loss(y, y_hat)

            total_loss += x.shape[0] * l.item()
            total_samples += x.shape[0]

    return total_loss / total_samples

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='train_ftm.py')
    parser.add_argument('--train', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--test', type=str, required=True, help="Path to test data directory.")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs to train.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--d_query', type=int, default=32, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--save_to', type=str, required=True, help="Set a file to save the models to.")
    opt = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data as a flat tensors
    train_data = VGMidi(opt.train, seq_len=opt.seq_len)
    test_data = VGMidi(opt.test, seq_len=opt.seq_len)

    # Compute vocab size
    vocab_size = max(train_data.vocab_size, test_data.vocab_size) + 1

    # Batchfy flat tensor data
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=True)

    # Build linear transformer
    model = MusicGenerator(n_tokens=vocab_size,
                            d_query=opt.d_query,
                            d_model=opt.d_query * opt.n_heads,
                            seq_len=opt.seq_len,
                           mixtures=3,
                     attention_type="causal-linear",
                           n_layers=opt.n_layers,
                           n_heads=opt.n_heads).to(device)

    # Train model
    trained_model = train(model, train_dataloader, test_dataloader, epochs=opt.epochs, lr=opt.lr, save_to=opt.save_to)
