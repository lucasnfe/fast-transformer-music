import torch
import time
import argparse
import numpy as np

from vgmidi import VGMidiLabelled

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import SGDClassifier
from sklearn.metrics                 import confusion_matrix

from models.music_emotion_classifier import MusicEmotionClassifier, MusicEmotionClassifierBaseline
from train_emotion_classifier import train

def encode_bow(xs, vocab_size):
    # Create bag of words
    vocabulary = {str(i):i for i in range(vocab_size)}

    # Make sure pieces are strings
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

def train_baseline(x_train, x_test, y_train, y_test):
    # Fit Logistic Regression
    clf = SGDClassifier(loss="log",
                 penalty="l2",
                   alpha=1e-4,
            random_state=42,
                max_iter=1000,
                     tol=0.001)

    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)

    accuracy, confusion = evaluate_clf(y_test, y_hat)
    print("Accuracy:", accuracy)
    print(confusion)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='generate.py')
    parser.add_argument('--train', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--test', type=str, required=True, help="Path to train data directory.")
    parser.add_argument('--vocab_size', type=int, required=True, help="Vocabulary size.")
    parser.add_argument('--epochs', type=int, default=100, help="Epochs to train.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--d_model', type=int, required=True, help="Model dimensions.")
    opt = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data as a flat tensors
    vgmidi_train = VGMidiLabelled(opt.train, seq_len=opt.seq_len, balance=True)
    vgmidi_test = VGMidiLabelled(opt.test, seq_len=opt.seq_len, balance=True)

    # Get training data
    x_train = encode_bow(vgmidi_train.get_pieces_txt(), opt.vocab_size)
    x_test = encode_bow(vgmidi_test.get_pieces_txt(), opt.vocab_size)

    train_baseline(x_train, x_test, vgmidi_train.labels, vgmidi_test.labels)

    # Batchfy flat tensor data
    train_loader = torch.utils.data.DataLoader(vgmidi_train, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(vgmidi_test, batch_size=opt.batch_size, shuffle=False)

    # Build linear transformer
    model = MusicEmotionClassifierBaseline(n_tokens=opt.vocab_size,
                                            d_model=opt.d_model).to(device)

    train(model, train_loader, test_loader, opt.epochs, opt.lr)
