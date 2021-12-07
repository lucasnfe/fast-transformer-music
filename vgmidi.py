import os
import torch
import csv

PAD_TOKEN = 389

import json
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

class VGMidiEmotion:
    def __init__(self, valence, arousal):
        assert valence in (-1, 0, 1)
        assert arousal in (-1, 0, 1)

        self.va = np.array([valence, arousal])
        self.quad2emotion = {0: "neutral",
                             1: "happy",
                             2: "angry",
                             3: "sad",
                             4: "calm"}

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
            return 1
        elif self.va[0] == -1 and self.va[1] == 1:
            return 2
        elif self.va[0] == -1 and self.va[1] == -1:
            return 3
        elif self.va[0] == 1 and self.va[1] == -1:
            return 4

        return 0

    @staticmethod
    def from_quadrant(quad):
        if quad == 0:
            return VGMidiEmotion(0, 0)
        elif quad == 1:
            return VGMidiEmotion(1, 1)
        elif quad == 2:
            return VGMidiEmotion(-1, 1)
        elif quad == 3:
            return VGMidiEmotion(-1,-1)
        elif quad == 4:
            return VGMidiEmotion(1,-1)

        return None

    @staticmethod
    def rand():
        va = np.random.rand(2)
        return SBBSEmotion(va[0], va[1])

    @staticmethod
    def load_signals(path):
        emotion_signals = {}
        with open(path) as f:
            emotion_signals = json.load(f)

        # Convert emotion arrays to SBBSEmotion instances
        for ep,signals in emotion_signals.items():
            for i in range(len(emotion_signals[ep])):
                ((valence, arousal), duration) = emotion_signals[ep][i]
                emotion_signals[ep][i] = (VGMidiEmotion(valence, arousal), duration)

        return emotion_signals

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
    def __init__(self, midi_csv, seq_len):
        self.seq_len = seq_len
        self.pieces = self._load_pieces(midi_csv)

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        return torch.tensor(self.pieces[idx])

    def _load_pieces(self, midi_csv):
        pieces = {}

        for row in csv.DictReader(open(midi_csv, "r")):
            filepath = row["midi"]
            piece_id = row["id"]
            valence, arousal = int(row["valence"]), int(row["arousal"])

            # Form midi filepath
            piecepath = os.path.join(os.path.dirname(midi_csv), filepath)

            # Form txt filepath
            txt_file = os.path.splitext(piecepath)[0] + ".txt"
            if os.path.exists(txt_file):
                emotion = VGMidiEmotion(valence, arousal)

                # Read txt file
                with open(txt_file) as fp:
                    tokens = fp.read().splitlines()[0]

                if piece_id not in pieces:
                    pieces[piece_id] = []

                # Append piece with emotion to the dataset
                pieces[piece_id].append((tokens, emotion))

        return pieces

    def get_emotion_data(self, train_size=.7, test_size=.3, remove_neutral=False):
        xs, ys, groups = [], [], []
        for group, piece_id in enumerate(sorted(self.pieces)):
            for phrase in self.pieces[piece_id]:
                tokens, emotion = phrase
                emotion = emotion.get_quadrant()
                if remove_neutral:
                    if emotion == 0:
                        continue
                    emotion -= 1

                xs.append(tokens)
                ys.append(emotion)
                groups.append(group)

        # Split data
        xs, ys, groups = np.array(xs), np.array(ys), np.array(groups)
        kfold = GroupShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size, random_state=42)
        for train_index, test_index in kfold.split(xs, ys, groups):
            x_train, x_test = xs[train_index], xs[test_index]
            y_train, y_test = ys[train_index], ys[test_index]

        return x_train, x_test, y_train, y_test
