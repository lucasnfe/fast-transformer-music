import torch
import numpy as np

END_TOKEN = 389

class PieceState():
    def __init__(self, sequence, i, memory, sequence_score=1e-9, terminal=False):
        self.sequence = sequence
        self.sequence_score = sequence_score

        self.i = i
        self.memory = memory
        self.terminal = terminal

    def find_children(self, language_model, top_k, max_len):
        # If the piece is finished then no tokens can be added
        if self.terminal:
            return None

        with torch.no_grad():
            print("i", self.i)
            print("sequence", self.sequence, self.sequence[self.i:self.i+1])

            x_i = self.sequence[self.i:self.i+1].unsqueeze(0)
            y_i, memory = language_model(x_i, i=self.i, memory=self.memory)
            top_probs, top_tokens = torch.topk(torch.softmax(y_i, dim=1), top_k, sorted=False, dim=1)

        children = []
        probabilities = {}
        for t,p in zip(top_tokens.squeeze(), top_probs.squeeze()):
            child = self.add_token(t, p, max_len, memory)
            children.append(child)
            probabilities[child] = p

        return set(children), probabilities

    def add_token(self, token, p, max_len, memory):
        sequence = torch.cat((self.sequence, token.unsqueeze(0)), dim=0)
        sequence_score = float(p)
        is_terminal = sequence.shape[-1] >= max_len or token == END_TOKEN
        return PieceState(sequence, self.i + 1, memory, sequence_score, is_terminal)

    def is_terminal(self):
        return self.terminal

    def __hash__(self):
        return hash(self.sequence)

    def __eq__(self, other):
        return (self.sequence == other.sequence).all()

    def __str__(self):
        return "[" + ", ".join([str(t) for t in self.sequence]) + "]"

    def __len__(self):
        return self.sequence.shape[-1]

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(self, language_model, temperature=1.0, top_k=0, max_len=256, c_puct=1,
                       emotion_classifier=None, emotion=None):

        self.Qsa = {} # stores Q values for s,a (as defined in the paper)
        self.Nsa = {} # stores #times edge s,a was visited
        self.Ps  = {} # stores language model policy
        self.Ns  = {}
        self.children = {}  # children of each node

        self.language_model = language_model
        self.emotion_classifier = emotion_classifier
        self.emotion = emotion

        self.top_k = top_k
        self.c_puct = c_puct
        self.max_len = max_len
        self.temperature = temperature

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f'choose called on terminal node {node}')

        C = [c for c in self.children[node]]
        N = np.array([self.Nsa[node][c] if c in self.Nsa[node] else 0 for c in self.children[node]])
        N = N**(1/self.temperature)

        print("C:", [c.sequence for c in C])
        print("N:", N/np.sum(N))
        sampled_child = np.random.choice(len(N), size=1, p=N/np.sum(N))[0]

        return C[sampled_child]

    def step(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        if node.is_terminal():
            value = self._reward(node)
            return value

        if node not in self.children:
            self.children[node], self.Ps[node] = node.find_children(self.language_model, self.top_k, self.max_len)
            self.Qsa[node] = {}
            self.Nsa[node] = {}
            self.Ns[node] = 0

            value = self._reward(node)
            return value

        leaf = self._select(node)
        value = self.step(leaf)

        if leaf in self.Qsa[node]:
            self.Qsa[node][leaf] = (self.Nsa[node][leaf] * self.Qsa[node][leaf] + value)/(self.Nsa[node][leaf] + 1)
            self.Nsa[node][leaf] += 1
        else:
            self.Qsa[node][leaf] = value
            self.Nsa[node][leaf] = 1

        self.Ns[node] += 1

        return value

    def _reward(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        with torch.no_grad():
            # Emotion score
            sequence = node.sequence.unsqueeze(0)
            emotion_score = torch.softmax(self.emotion_classifier(sequence), dim=1)
            emotion_score = float(emotion_score.squeeze()[self.emotion])

            # Language score
            # language_score = node.sequence_score/len(node.sequence)

        # reward = language_score
        print("reward:", emotion_score)
        return emotion_score

    def _select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        def puct(c):
            if c not in self.Qsa[node]:
                return float("inf")

            "Upper confidence bound for trees"
            return self.Qsa[node][c] + self.c_puct * self.Ps[node][c] * np.sqrt(self.Ns[node])/(
                    1 + self.Nsa[node][c])

        return max(self.children[node], key=puct)
