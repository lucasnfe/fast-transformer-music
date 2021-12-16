import os
import torch
import csv

PAD_TOKEN = 390

import json
import numpy as np

from sklearn.model_selection import GroupKFold

class VGMidiEmotion:
    def __init__(self, valence, arousal):
        assert valence in (-1, 1)
        assert arousal in (-1, 1)

        self.va = np.array([valence, arousal])
        self.quad2emotion = {0: "happy", 1: "angry", 2: "sad", 3: "calm"}

    def __eq__(self, other):
        if other == None:
            return False
        return (self.va == other.va).all()

    def __ne__(self, other):
        if other == None:
            return True
        return (self.va != other.va).any()

    def __str__(self):
        return self.quad2emotion[self.get_quadrant()]

    def __getitem__(self, key):
        return self.va[key]

    def __setitem__(self, key, value):
        self.va[key] = value

    def get_quadrant(self):
        if self.va[0] == 1 and self.va[1] == 1:
            return 0
        elif self.va[0] == -1 and self.va[1] == 1:
            return 1
        elif self.va[0] == -1 and self.va[1] == -1:
            return 2
        elif self.va[0] == 1 and self.va[1] == -1:
            return 3

        return None

class VGMidiUnlabelled(torch.utils.data.Dataset):
    def __init__(self, pieces_path, seq_len):
        self.seq_len = seq_len
        self.pieces = self._load_pieces(pieces_path, seq_len)

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        x = self.pieces[idx][:-1]
        y = self.pieces[idx][1:]
        return torch.tensor(x), torch.tensor(y)

    @property
    def vocab_size(self):
        vocab_size = float("-inf")
        for piece in self.pieces:
            vocab_size = max(vocab_size, max(piece))
        return vocab_size

    def _load_txt(self, file_path):
        loaded_list = []
        with open(file_path) as f:
            loaded_list = [int(token) for token in f.read().split()]
        return loaded_list

    def _load_pieces(self, pieces_path, seq_len):
        pieces = []
        for file_path in os.listdir(pieces_path):
            full_path = os.path.join(pieces_path, file_path)
            if os.path.isfile(full_path):
                # Make sure the piece has been encoded into a txt file (see encoder.py)
                file_name, extension = os.path.splitext(file_path)
                if extension.lower() == ".txt":
                    # Load entire piece
                    encoded = self._load_txt(full_path)

                    # Split the piece into sequences of len seq_len
                    for i in range(0, len(encoded), seq_len):
                        piece_seq = encoded[i:i+seq_len]
                        if len(piece_seq) < seq_len:
                            # Pad sequence
                            piece_seq += [PAD_TOKEN] * (seq_len - len(piece_seq))

                        pieces.append(piece_seq)

        return pieces

class VGMidiLabelled(torch.utils.data.Dataset):
    def __init__(self, midi_csv, seq_len, balance=False, prefix=0):
        self.seq_len = seq_len
        pieces, labels, groups = self._load_pieces(midi_csv, seq_len)

        # Generate prefixes of different sizes
        if prefix > 0:
            pieces, labels, groups = self._gen_prefixes(pieces, labels, groups, prefix)

        self.pieces = pieces
        self.labels = labels
        self.groups = groups

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        return torch.tensor(self.pieces[idx]), torch.tensor(self.labels[idx])

    def _load_txt(self, file_path):
        loaded_list = []
        with open(file_path) as f:
            loaded_list = [int(token) for token in f.read().split()]
        return loaded_list

    def _load_pieces(self, midi_csv, seq_len):
        pieces = []
        labels = []
        groups = []

        csv_dir, csv_name = os.path.split(midi_csv)
        for row in csv.DictReader(open(midi_csv, "r")):
            piece_path = os.path.join(csv_dir, row["midi"])

            # Make sure the piece has been encoded into a txt file (see encoder.py)
            file_name, extension = os.path.splitext(piece_path)
            if os.path.isfile(file_name + ".txt"):

                # Load entire piece
                encoded = self._load_txt(file_name + ".txt")

                # Get emotion
                emotion = VGMidiEmotion(int(row["valence"]), int(row["arousal"]))

                # Split the piece into sequences of len seq_len
                for i in range(0, len(encoded), seq_len):
                    piece_seq = encoded[i:i+seq_len]
                    if len(piece_seq) < seq_len:
                        # Pad sequence
                        piece_seq += [PAD_TOKEN] * (seq_len - len(piece_seq))

                    pieces.append(piece_seq)
                    labels.append(emotion.get_quadrant())
                    groups.append(row["game"])

        assert len(pieces) == len(labels) == len(groups)
        return np.array(pieces), np.array(labels), np.array(groups)

    def _gen_prefixes(self, xs, ys, groups, prefix_step):
        # Generate prefixes
        x_prefixes = []
        y_prefixes = []
        groups_prefixes = []

        for x,y,g in zip(xs, ys, groups):
            for prefix_size in range(prefix_step, len(x) + prefix_step, prefix_step):
                prefix = list(x[:prefix_size])
                if len(prefix) < self.seq_len:
                    # Pad sequence
                    prefix += [PAD_TOKEN] * (self.seq_len - len(prefix))

                x_prefixes.append(prefix)
                y_prefixes.append(y)
                groups_prefixes.append(g)

        return np.array(x_prefixes), np.array(y_prefixes), np.array(groups_prefixes)

    def get_pieces_txt(self):
        pieces = []
        for piece in self.pieces:
            piece_txt = " ".join([str(token) for token in piece])
            pieces.append(piece_txt)

        return pieces
