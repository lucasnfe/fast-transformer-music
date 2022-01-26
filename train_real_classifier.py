import torch
import time
import argparse
import numpy as np

from vgmidi import VGMidiLabelled, VGMidiRealFake, VGMidiSampler, pad_collate
from sklearn.metrics import confusion_matrix
from models.music_emotion_classifier import MusicEmotionClassifier, MusicEmotionClassifierBaseline

# Reproducitility
np.random.seed(42)
torch.manual_seed(42)

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

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Train model for one epoch
        train_step(model, train_data, epoch, lr, criterion, optimizer)

        # Evaluate model on test set
        val_accuracy, confusion = evaluate(model, test_data)

        elapsed = time.time() - epoch_start_time

        # Log training statistics for this epoch
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'accuracy {val_accuracy:5.2f} | best val accuracy: {best_val_accuracy:5.2f} ')

        # Save best model so far
        if val_accuracy > best_val_accuracy:
            print(f'Validation accuracy improved from {best_val_accuracy:5.2f} to {val_accuracy:5.2f}.'
                  f'Saving model to {save_to}.')

            print(confusion)

            best_val_accuracy = val_accuracy
            save_model(model, optimizer, epoch, save_to)

        print('-' * 89)

def train_step(model, train_data, epoch, lr, criterion, optimizer, log_interval=50):
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
        loss = criterion(y_hat.view(-1), y.view(-1))
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

    ys = []
    ys_hat = []

    with torch.no_grad():
        for batch, (x, y) in enumerate(test_data):
            x = x.to(device)
            y = y.to(device)

            # Evaluate
            y_hat = model(x)

            # the class with the highest energy is what we choose as prediction
            y_hat = torch.round(torch.sigmoid(y_hat))

            ys += y.tolist()
            ys_hat += y_hat.tolist()

    accuracy = np.mean(np.array(ys) == np.array(ys_hat))
    confusion = confusion_matrix(ys, ys_hat)

    return accuracy, confusion

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='train_emotion_classifier.py')
    parser.add_argument('--train', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--test', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--train_fake', type=str, required=True, help="Path to fake train data directory.")
    parser.add_argument('--test_fake', type=str, required=True, help="Path to fake test data directory.")
    parser.add_argument('--pre_trained', type=str, required=False, help="Path to load model from.")
    parser.add_argument('--vocab_size', type=int, required=True, help="Vocabulary size.")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs to train.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--n_finetune', type=int, default=-1, help="Index of last layer to finetune.")
    parser.add_argument('--d_query', type=int, default=32, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--prefix', type=int, default=0, help="Traing with different prefix sizes.")
    parser.add_argument('--save_to', type=str, required=True, help="Set a file to save the models to.")
    opt = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load real data
    train_real_data = VGMidiRealFake(opt.train, opt.seq_len, label=1.0, prefix=opt.prefix)
    test_real_data = VGMidiRealFake(opt.test, opt.seq_len, label=1.0, prefix=opt.prefix)

    # Load fake data
    train_fake_data = VGMidiRealFake(opt.train_fake, seq_len=opt.seq_len, label=0.0, prefix=opt.prefix)
    test_fake_data = VGMidiRealFake(opt.test_fake, seq_len=opt.seq_len, label=0.0, prefix=opt.prefix)

    # Mix real and fake data
    train_data = torch.utils.data.ConcatDataset((train_real_data, train_fake_data))
    test_data = torch.utils.data.ConcatDataset((test_real_data, test_fake_data))

    # Batchfy flat tensor data
    train_loader = torch.utils.data.DataLoader(train_data,
                             batch_size=opt.batch_size,
                                sampler=VGMidiSampler(train_data, bucket_size=opt.prefix, max_len=opt.seq_len, shuffle=True),
                             collate_fn=pad_collate)

    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=opt.batch_size,
                                               sampler=VGMidiSampler(test_data, bucket_size=opt.prefix, max_len=opt.seq_len, shuffle=False),
                                            collate_fn=pad_collate)

    # Build linear transformer
    model = MusicEmotionClassifier(n_tokens=opt.vocab_size,
                            d_query=opt.d_query,
                            d_model=opt.d_query * opt.n_heads,
                            seq_len=opt.seq_len,
                     attention_type="linear",
                           n_layers=opt.n_layers,
                            n_heads=opt.n_heads).to(device)

    # Load model
    if opt.pre_trained:
        print(f'Fine-tuning model {opt.pre_trained}')
        model.load_state_dict(torch.load(opt.pre_trained, map_location=device)["model_state"])

        # Lock paramters and reset last l
        for i, layer in enumerate(model.transformer.layers):
            if i < opt.n_finetune:
                for param in layer.parameters():
                    param.requires_grad = False

    # Add classification head
    model = torch.nn.Sequential(model,
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(opt.vocab_size, 1)).to(device)

    train(model, train_loader, test_loader, opt.epochs, opt.lr, opt.save_to)
