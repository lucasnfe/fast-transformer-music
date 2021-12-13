import torch
import time
import argparse
import numpy as np

from vgmidi import VGMidiLabelled

from models.music_emotion_classifier import MusicEmotionClassifier, MusicEmotionClassifierBaseline

def save_model(model, optimizer, epoch, save_to):
    model_path = save_to.format(epoch)
    torch.save(
        dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch
        ),
        model_path)

def train(model, train_data, test_data, epochs, lr, save_to):
    model.train()

    best_model = None
    best_val_accuracy = 0

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Train model for one epoch
        train_step(model, train_data, epoch, lr, criterion, optimizer)

        # Evaluate model on test set
        val_accuracy = evaluate(model, test_data)

        elapsed = time.time() - epoch_start_time

        # Log training statistics for this epoch
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'accuracy {val_accuracy:5.2f}')

        # Save best model so far
        if val_accuracy > best_val_accuracy:
            print(f'Validation accuracy improved from {best_val_accuracy:5.2f} to {val_accuracy:5.2f}.'
                  f'Saving model to {save_to}.')

            best_val_accuracy = val_accuracy
            save_model(model, optimizer, epoch, save_to)

        print('-' * 89)

def train_step(model, train_data, epoch, lr, criterion, optimizer, log_interval=1):
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
        loss = criterion(y_hat[:,-1,:], y)
        loss.backward()
        optimizer.step()

        # print statistics
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

    print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
          f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
          f'loss {cur_loss:5.2f}')

def evaluate(model, test_data):
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_data):
            x = x.to(device)
            y = y.to(device)

            # Evaluate
            y_hat = model(x)

            # the class with the highest energy is what we choose as prediction
            _, y_hat = torch.max(y_hat[:,-1,:].data, dim=1)

            total += y.size(0)
            correct += (y_hat == y).sum().item()

    return 100 * correct / total

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='generate.py')
    parser.add_argument('--train', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--test', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--model', type=str, required=True, help="Path to load model from.")
    parser.add_argument('--vocab_size', type=int, required=True, help="Vocabulary size.")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs to train.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--d_query', type=int, default=32, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    opt = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data as a flat tensors
    vgmidi_train = VGMidiLabelled(opt.train, seq_len=opt.seq_len)
    vgmidi_test = VGMidiLabelled(opt.test, seq_len=opt.seq_len)

    # Batchfy flat tensor data
    train_loader = torch.utils.data.DataLoader(vgmidi_train, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(vgmidi_test, batch_size=opt.batch_size, shuffle=False)

    # Build linear transformer
    model = MusicEmotionClassifier(n_tokens=opt.vocab_size,
                            d_query=opt.d_query,
                            d_model=opt.d_query * opt.n_heads,
                            seq_len=opt.seq_len,
                     attention_type="linear",
                           n_layers=opt.n_layers,
                            n_heads=opt.n_heads).to(device)

    # Load model
    model.load_state_dict(torch.load(opt.model, map_location=device)["model_state"])

    # Lock paramters and reset last l
    for param in model.parameters():
        param.requires_grad = False

    # Add classification head
    model = torch.nn.Sequential(model, torch.nn.Linear(opt.vocab_size, 4)).to(device)

    train(model, train_loader, test_loader, opt.epochs, opt.lr, opt.save_to)
