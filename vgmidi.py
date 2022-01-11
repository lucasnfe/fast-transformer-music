import os
import torch
import csv
import json
import numpy as np

from sklearn.model_selection import GroupKFold

PAD_TOKEN = 390

class VGMidiEmotion:
    def __init__(self, valence, arousal):
        assert valence in (-1, 1)
        assert arousal in (-1, 1)

        self.va = np.array([valence, arousal])
        self.quad2emotion = {0: "q1", 1: "q2", 2: "q3", 3: "q4"}

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

def pad_collate(batch):
    max_len = max([example[0].shape[-1] for example in batch])

    padded_examples = []
    targets = []

    for example in batch:
        x, y = example

        padding_len = max_len - x.shape[-1]
        if padding_len > 0:
            padding = torch.full((padding_len,), PAD_TOKEN)
            x = torch.cat((x, padding), dim=0)

        padded_examples.append(x)
        targets.append(y)

    return torch.stack(padded_examples), torch.stack(targets)

class VGMidiSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, bucket_size=64, max_len=2048, shuffle=False):
        self.max_len = max_len
        self.bucket_size = bucket_size
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        buckets = [[] for i in range(0, self.max_len, self.bucket_size)]

        for i, example in enumerate(self.data_source.pieces):
            bucket_ix = len(example) // self.bucket_size - 1
            buckets[bucket_ix].append(i)

        idxs = []
        for i in range(len(buckets)):
            if self.shuffle:
                np.random.shuffle(buckets[i])
            idxs += buckets[i]

        return iter(idxs)

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

                # Time encoded piece to max len
                encoded = encoded[:seq_len]

                # Pad sequence
                # if len(encoded) < seq_len:
                #     encoded += [PAD_TOKEN] * (seq_len - len(encoded))

                # Get emotion
                emotion = VGMidiEmotion(int(row["valence"]), int(row["arousal"]))

                pieces.append(encoded)
                labels.append(emotion.get_quadrant())
                groups.append(row["game"])

        assert len(pieces) == len(labels) == len(groups)
        return pieces, labels, groups

    def _gen_prefixes(self, xs, ys, groups, prefix_step):
        # Generate prefixes
        x_prefixes = []
        y_prefixes = []
        groups_prefixes = []

        for x,y,g in zip(xs, ys, groups):
            for prefix_size in range(prefix_step, len(x) + prefix_step, prefix_step):
                prefix = list(x[:prefix_size])
                # if len(prefix) < self.seq_len:
                #     # Pad sequence
                #     prefix += [PAD_TOKEN] * (self.seq_len - len(prefix))

                x_prefixes.append(prefix)
                y_prefixes.append(y)
                groups_prefixes.append(g)

        return x_prefixes, y_prefixes, groups_prefixes

    def get_pieces_txt(self):
        pieces = []
        for piece in self.pieces:
            piece_txt = " ".join([str(token) for token in piece])
            pieces.append(piece_txt)

        return pieces
