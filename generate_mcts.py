import os
import copy
import json
import torch
import argparse
import numpy as np

from mcts import MCTS#, PieceState
from encoder import decode_midi, decode_events

from models.music_generator import MusicGenerator
from models.music_emotion_classifier import MusicEmotionClassifier

START_TOKEN = 388

def load_language_model(model, vocab_size, d_query, n_layers, n_heads, seq_len):
    language_model = MusicGenerator(n_tokens=vocab_size,
                                     d_query=d_query,
                                     d_model=d_query * n_heads,
                                     seq_len=seq_len,
                              attention_type="causal-linear",
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

def generate(language_model, emotion_classifier, emotion, seq_len, vocab_size, piece, roll_steps=30, temperature=1.0, k=0, c=3.0):
    tree = MCTS(language_model,
                emotion_classifier,
                emotion,
                vocab_size,
                device,
                seq_len,
                temperature, k, c)

    # Init mucts
    try:
        while True:
            print("Current piece:", piece)

            for step in range(roll_steps):
                print("Rollout: %d" % step)
                tree.step(piece, prob=1e-8)

            # Choose next state
            token = tree.choose(piece)
            piece = tree._get_next_state(piece, token)

            if tree._is_terminal(piece):
                break

    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt.")

    return [int(token) for token in piece[-1]]

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='genrate_mcts.py')
    parser.add_argument('--lm', type=str, required=True, help="Path to load language model from.")
    parser.add_argument('--clf', type=str, required=True, help="Path to load emotion classifier from.")
    parser.add_argument('--emotion', type=int, required=True, help="Piece emotion.")
    parser.add_argument('--k', type=int, default=0, help="Number k of elements to consider while sampling.")
    parser.add_argument('--c', type=float, default=1.0, help="Constant c for puct.")
    parser.add_argument('--vocab_size', type=int, required=True, help="Vocabulary size.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of transformer layers.")
    parser.add_argument('--d_query', type=int, default=32, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--save_to', type=str, required=True, help="Set a file to save the models to.")
    parser.add_argument('--device', type=str, required=False, help="Force device.")
    opt = parser.parse_args()

    # Set up torch device
    if opt.device:
        device = torch.device(opt.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    language_model = load_language_model(opt.lm, opt.vocab_size, opt.d_query, opt.n_layers, opt.n_heads, opt.seq_len)
    emotion_classifier = load_emotion_classifier(opt.clf, opt.vocab_size, opt.d_query, opt.n_layers, opt.n_heads, opt.seq_len)

    # Define prime sequence
    prime = [START_TOKEN]
    prime = torch.tensor(prime).unsqueeze(dim=0).to(device)

    piece = generate(language_model, emotion_classifier, opt.emotion, 512, opt.vocab_size, prime, k=opt.k, c=opt.c)
    decode_midi(piece, opt.save_to)
    print(piece)