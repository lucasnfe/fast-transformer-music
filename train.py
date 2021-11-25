import os
import time
import math
import copy
import torch
import argparse

from vgmidi import VGMidi
from model import MusicGenerator

def train(model, train_data, test_data, epochs=100, lr=0.001):
    best_model = None
    best_val_loss = float('inf')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Train model for one epoch
        train_step(model, train_data, epoch, lr, criterion, optimizer)

        # Evaluate model on test set
        val_loss = evaluate(model, test_data, criterion)

        elapsed = time.time() - epoch_start_time

        # Compute validation perplexity
        val_ppl = math.exp(val_loss)

        # Log training statistics for this epoch
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

    return best_model

def train_step(model, train_data, epoch, lr, criterion, optimizer, log_interval=100):
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
        loss = criterion(y_hat.view(-1, vocab_size), y)
        loss.backward()
        optimizer.step()

        # Log training statistics
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            log_stats(optimizer, epoch, batch, len(train_data), total_loss, start_time, log_interval)
            total_loss = 0
            start_time = time.time()

def log_stats(optimizer, epoch, batch, num_batches, total_loss, start_time, log_interval):
    # Get current learning rate
    lr = optimizer.param_groups[0]['lr']

    # Compute duration of each batch
    ms_per_batch = (time.time() - start_time) * 1000 / log_interval

    # Compute current loss
    cur_loss = total_loss / log_interval

    # compute current perplexity
    ppl = math.exp(cur_loss)

    print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
          f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
          f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')

def evaluate(model, test_data, criterion):
    model.eval()

    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_data):
            x = x.to(device)
            y = y.to(device)

            # Evaluate
            y_hat = model(x)
            loss = criterion(y_hat.view(-1, vocab_size), y)

            total_loss += x.shape[0] * loss.item()
            total_samples += x.shape[0]

    return total_loss / total_samples

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='train_ftm.py')
    parser.add_argument('--train', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--test', type=str, required=True, help="Path to test data directory.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs to train.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--d_query', type=int, default=32, help="Dimension of the query matrix.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
    opt = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data as a flat tensors
    train_data = VGMidi(opt.train, seq_len=opt.seq_len)
    test_data = VGMidi(opt.test, seq_len=opt.seq_len)

    # Compute vocab size
    vocab_size = max(train_data.vocab_size, test_data.vocab_size) + 1

    # Batchfy flat tensor data
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    # Build linear transformer
    model = MusicGenerator(n_tokens=vocab_size,
                                d_model=opt.d_query * opt.n_heads,
                                seq_len=opt.seq_len,
                         attention_type="causal-linear",
                               n_layers=opt.n_layers,
                                n_heads=opt.n_heads).to(device)

    # Train model
    trained_model = train(model, train_data, test_data, epochs=opt.epochs, lr=opt.lr)
