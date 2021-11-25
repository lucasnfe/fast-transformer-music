import os
import torch

PAD_TOKEN = 389

class VGMidi(torch.utils.data.Dataset):
    def __init__(self, pieces_path, seq_len):
        self.seq_len = seq_len
        self.pieces = self._load_pieces(pieces_path, seq_len)

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        x = self.pieces[idx][:-1]
        y = self.pieces[idx][1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

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
