import os
import json
import torch
import argparse

from mcts import MCTS, PieceState
from encoder import decode_midi

from models.music_generator_recurrent import RecurrentMusicGenerator
from models.music_emotion_classifier import MusicEmotionClassifier

START_TOKEN = 388

def load_language_model(model, vocab_size, d_query, n_layers, n_heads, seq_len):
    language_model = RecurrentMusicGenerator(n_tokens=vocab_size,
                                              d_query=d_query,
                                              d_model=d_query * n_heads,
                                              seq_len=seq_len,
                                       attention_type="linear",
                                             n_layers=n_layers,
                                              n_heads=n_heads).to(device)

    # Load model
    language_model.load_state_dict(torch.load(model, map_location=device)["model_state"])
    language_model.eval()

    return language_model

def load_emotion_classifier(model, vocab_size, d_query, n_layers, n_heads, seq_len):
    # Load Emotion Classifier
    emotion_classifier = MusicEmotionClassifier(n_tokens=vocab_size,
                                                 d_query=d_query,
                                                 d_model=d_query * n_heads,
                                                 seq_len=seq_len,
                                          attention_type="linear",
                                                n_layers=n_layers,
                                                 n_heads=n_heads).to(device)

    emotion_classifier = torch.nn.Sequential(emotion_classifier,
                         torch.nn.Dropout(0.0),
                         torch.nn.Linear(vocab_size, 4)).to(device)

    emotion_classifier.load_state_dict(torch.load(model, map_location=device)["model_state"])
    emotion_classifier.eval()

    return emotion_classifier

def generate(language_model, emotion_classifier, emotion, seq_len, prime, roll_steps=100, c_uct=1.0, k=10, temperature=1.0):
    tree = MCTS(language_model,
                 temperature,
                 k,
                 seq_len,
                 c_uct,
                 emotion_classifier,
                 emotion)

    # Init mucts
    piece = PieceState(prime, i=0, memory=None)

    try:
        while True:
            print("Current piece:", piece, piece.i)

            for step in range(roll_steps):
                print("Rollout: %d" % step)
                tree.step(piece)

            piece = tree.choose(piece)

            if piece.terminal:
                break

    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt.")

    return [int(token) for token in piece.sequence]

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='genrate_mcts.py')
    parser.add_argument('--lm', type=str, required=True, help="Path to load language model from.")
    parser.add_argument('--clf', type=str, required=True, help="Path to load emotion classifier from.")
    parser.add_argument('--emotion', type=int, required=True, help="Piece emotion.")
    parser.add_argument('--k', type=int, default=0, help="Number k of elements to consider while sampling.")
    parser.add_argument('--vocab_size', type=int, required=True, help="Vocabulary size.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--d_query', type=int, default=32, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    opt = parser.parse_args()

    # Set up torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    language_model = load_language_model(opt.lm, opt.vocab_size, opt.d_query, opt.n_layers, opt.n_heads, opt.seq_len)
    emotion_classifier = load_emotion_classifier(opt.clf, opt.vocab_size, opt.d_query, opt.n_layers, opt.n_heads, opt.seq_len)

    # Define prime sequence
    prime = [START_TOKEN]
    prime = torch.tensor(prime, dtype=torch.int64).to(device)

    piece = generate(language_model, emotion_classifier, opt.emotion, opt.seq_len, prime)
    decode_midi(piece, "results/generated_piece_mcts.mid")
    print(piece)
