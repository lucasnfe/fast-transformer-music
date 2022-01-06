import torch
import numpy as np
import plotext as plt

from generate import filter_top_k

END_TOKEN = 389

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(self, language_model, vocab_size, temperature=1.0, top_k=0, max_len=512, c_puct=1,
                       emotion_classifier=None, emotion=None):

        self.Qsa = {} # stores Q values for s,a (as defined in the paper)
        self.Nsa = {} # stores #times edge s,a was visited
        self.Ps  = {} # stores language model policy
        self.Ns  = {}

        self.language_model = language_model
        self.emotion_classifier = emotion_classifier
        self.emotion = emotion
        self.vocab_size = vocab_size

        self.top_k = top_k
        self.c_puct = c_puct
        self.max_len = max_len
        self.temperature = temperature

    def diff_distros(self, old, new):
        tokens = [i for i in range(self.vocab_size)]

        plt.clf()
        plt.subplots(1, 2)

        plt.subplot(1, 1)
        plt.clc()
        plt.ylim(0.0,1.0)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        plt.title("Old token distribution")
        plt.plot(tokens, np.array(old, dtype=np.float64), marker='dot')

        plt.subplot(1, 2)
        plt.clc()
        plt.ylim(0.0,1.0)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        plt.title("New token distribution")
        plt.plot(tokens, np.array(new, dtype=np.float64), marker='dot')

        plt.show()

    def choose(self, state):
        "Choose the best successor of node. (Choose a move in the game)"
        s = self._get_string_representation(state)

        N = np.array([self.Nsa[(s, token)] if (s, token) in self.Nsa else 0 for token in range(self.vocab_size)])
        N = N**(1./self.temperature)
        N = N/np.sum(N)

        self.diff_distros(self.Ps[s].numpy(), N)

        random_idx = np.random.choice(len(N), p=N)
        # random_idx = torch.multinomial(self.Ps[s], num_samples=1)
        return random_idx

    def _get_next_state(self, state, token):
        return torch.cat((state, torch.tensor([[token]])), dim=1)

    def _is_terminal(self, state):
        return state.shape[-1] >= self.max_len or int(state[-1,-1]) == END_TOKEN

    def _get_string_representation(self, state):
        return " ".join([str(int(token)) for token in state[-1]])

    def step(self, state, prob):
        s = self._get_string_representation(state)

        "Make the tree one layer better. (Train for one iteration.)"
        if self._is_terminal(state):
            value = self._reward(state, prob)
            return value

        if s not in self.Ps:
            # leaf node
            self.Ps[s] = self._expand(state)
            self.Ns[s] = 0

            value = self._reward(state, prob)
            return value

        # Select next token
        token = self._select(s)

        # Recursevily call step until a leaf node is found
        next_state = self._get_next_state(state, token)

        print("\t selected:", token)
        value = self.step(next_state, prob + np.log(self.Ps[s][token]))

        if (s, token) in self.Qsa:
            self.Qsa[(s, token)] = (self.Nsa[(s, token)] * self.Qsa[(s, token)] + value) / (self.Nsa[(s, token)] + 1)
            self.Nsa[(s, token)] += 1
        else:
            self.Qsa[(s, token)] = value
            self.Nsa[(s, token)] = 1

        self.Ns[s] += 1

        return value

    def _expand(self, state):
        with torch.no_grad():
            print("\t expand:", state)

            # Compute logits
            # y_i, _ = self.language_model(sequence[:,i:i+1], i=i, memory=memory)
            y_i = self.language_model(state)[:,-1,:]

            # Filter top_k tokens
            y_i = filter_top_k(y_i, self.top_k)

            # Compute probability of the top_k token
            y_i = torch.softmax(y_i, dim=1)

        return y_i.squeeze()

    def _reward(self, state, prob):
        "Returns the reward for a random simulation (to completion) of `node`"
        with torch.no_grad():
            # Emotion score
            emotion_score = torch.softmax(self.emotion_classifier(state), dim=1)
            emotion_score = float(emotion_score.squeeze()[self.emotion])

        # reward = language_score
        print("\t reward:", emotion_score)
        return emotion_score

    def _select(self, s, eps=1e-8):
        "Upper confidence bound for trees"
        cur_best = -float('inf')
        best_token = -1

        for token in range(self.vocab_size):
            # Check if if is a valid move
            if self.Ps[s][token] > 0:
                if (s, token) in self.Qsa:
                    u = self.Qsa[(s, token)] + self.c_puct * self.Ps[s][token] * np.sqrt(self.Ns[s])/(
                            1 + self.Nsa[(s, token)])
                else:
                    u = self.c_puct * self.Ps[s][token] * np.sqrt(self.Ns[s] + eps)

                if u > cur_best:
                    cur_best = u
                    best_token = token

        return best_token
