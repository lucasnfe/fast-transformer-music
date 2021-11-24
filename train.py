import os
import time
import math
import copy
import torch
import argparse

from model import MusicGenerator

def _load_txt(file_path):
    loaded_list = []
    with open(file_path) as f:
        loaded_list = [int(token) for token in f.read().split()]

    return loaded_list

def load_txt_dir(dir_path):
    data = []
    for file_path in os.listdir(dir_path):
        full_path = os.path.join(dir_path, file_path)
        if os.path.isfile(full_path):
            file_name, extension = os.path.splitext(file_path)
            if extension.lower() == ".txt":
                encoded = _load_txt(full_path)
                data += encoded

    return data

def batchfy_data(data, batch_size):
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train(model, train_data, test_data, bptt, vocab_size, epochs=100, lr=0.001):
    best_val_loss = float('inf')
    best_model = None

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        train_step(model, train_data, epoch, bptt, vocab_size, lr, criterion, optimizer, scheduler)

        val_loss = evaluate(model, test_data, bptt, vocab_size, criterion)
        val_ppl = math.exp(val_loss)

        elapsed = time.time() - epoch_start_time

        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        scheduler.step()

    return best_model

def train_step(model, train_data, epoch, bptt, vocab_size, lr, criterion, optimizer, scheduler, log_interval=100):
    model.train()
    total_loss = 0.0

    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)

        # Forward pass
        output = model(data)
        loss = criterion(output.view(-1, vocab_size), targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Log training statistics
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)

            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')

            total_loss = 0
            start_time = time.time()

def evaluate(model, test_data, bptt, vocab_size, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in range(0, test_data.size(0) - 1, bptt):
            data, targets = get_batch(test_data, i, bptt)
            output = model(data)
            output_flat = output.view(-1, vocab_size)
            batch_size = data.size(0)
            total_loss += batch_size * criterion(output_flat, targets).item()

    return total_loss / (len(test_data) - 1)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='train_ftm.py')
    parser.add_argument('--train', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--test', type=str, required=True, help="Path to test data directory.")
    opt = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data as a flat tensors
    train_data = load_txt_dir(opt.train)
    test_data = load_txt_dir(opt.test)

    # Compute vocab size
    vocab_size = max(train_data + test_data) + 1

    # Batchfy flat tensor data
    train_data = batchfy_data(torch.tensor(train_data, dtype=torch.long), batch_size=32)
    test_data = batchfy_data(torch.tensor(test_data, dtype=torch.long), batch_size=32)

    # Build linear transformer
    model = MusicGenerator(n_tokens=vocab_size,
                                d_model=256,
                                seq_len=2048,
                         attention_type="causal-linear",
                               n_layers=2,
                                n_heads=8).to(device)

    # Train model
    trained_model = train(model, train_data, test_data, bptt=2048, vocab_size=vocab_size)
