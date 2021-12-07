import torch
import argparse
import numpy as np

from radam import RAdam
from vgmidi import VGMidiLabelled

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import SGDClassifier
from sklearn.metrics                 import confusion_matrix

from models.music_generator import MusicGenerator

def encode_bow(xs, vocab_size):
    # Create bag of words
    vocabulary = {str(i):i for i in range(vocab_size)}

    count_vect = CountVectorizer(vocabulary=vocabulary)
    xs_count = count_vect.fit_transform(xs)

    # Compute tfidf
    tfidf_transformer = TfidfTransformer()
    xs_tfidf = tfidf_transformer.fit_transform(xs_count).toarray()

    return xs_tfidf

def evaluate_clf(y_test, y_hat):
    accuracy = np.mean(y_test == y_hat)
    confusion = confusion_matrix(y_test, y_hat)
    return accuracy, confusion

def train_baseline(train_data, test_data):
    x_train, y_train = train_data
    y_train, y_test = test_data

    # Fit Logistic Regression
    clf = SGDClassifier(loss="log",
                 penalty="l1",
                   alpha=1e-3,
            random_state=42,
                max_iter=1000,
                     tol=0.001)

    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)

    accuracy, confusion = evaluate_clf(y_test, y_hat)
    print("Accuracy:", accuracy)
    print(confusion)

def train(model, train_data, test_data, epochs, lr):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        for batch, (x, y) in enumerate(train_data):
            # Forward pass
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)

            # Backward pass
            optimizer.zero_grad()
            loss = criterion(y_hat.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='generate.py')
    parser.add_argument('--train', type=str, required=True, help="Path to train data directory.")
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
    vgmidi_dataset = VGMidiLabelled(opt.train, seq_len=opt.seq_len)

    # Get training data
    x_train, x_test, y_train, y_test = vgmidi_dataset.get_emotion_data(remove_neutral=True)

    x_train = encode_bow(x_train, opt.vocab_size)
    x_test = encode_bow(x_test, opt.vocab_size)

    train_baseline((x_train, x_test), (y_train, y_test))

    # Build linear transformer
    model = MusicGenerator(n_tokens=opt.vocab_size,
                            d_query=opt.d_query,
                            d_model=opt.d_query * opt.n_heads,
                            seq_len=opt.seq_len,
                     attention_type="causal-linear",
                           n_layers=opt.n_layers,
                           n_heads=opt.n_heads).to(device)

    # Load model
    model.load_state_dict(torch.load(opt.model, map_location=device)["model_state"])
    model.eval()

    train(model, (x_train, x_test), (y_train, y_test), opt.epochs, opt.lr)
