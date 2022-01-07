import torch
import numpy as np
import plotext as plt

from generate import filter_top_k

END_TOKEN = 389

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    def __init__(self, language_model, emotion_classifier, emotion, vocab_size, device, seq_len=512, temperature=1.0, k=0, c=1):

        self.Qsa = {} # stores Q values for s,a (as defined in the paper)
        self.Nsa = {} # stores #times edge s,a was visited
        self.Ps  = {} # stores language model policy
        self.Ns  = {}

        self.language_model = language_model
        self.emotion_classifier = emotion_classifier
        self.emotion = emotion
        self.device = device

        self.k = k
        self.c = c
        self.seq_len = seq_len
        self.vocab_size = vocab_size
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

        N = self.Nsa[s]**(1./self.temperature)
        N = N/torch.sum(N)

        self.diff_distros(self.Ps[s].cpu().numpy(), N.cpu().numpy())

        random_idx = torch.multinomial(N, num_samples=1).squeeze()
        return random_idx

    def _get_next_state(self, state, token):
        return torch.cat((state, token.unsqueeze(0).unsqueeze(0)), dim=1)

    def _is_terminal(self, state):
        return state.shape[-1] >= self.seq_len or int(state[-1,-1]) == END_TOKEN

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

            self.Qsa[s] = torch.zeros(self.vocab_size).to(self.device)
            self.Nsa[s] = torch.zeros(self.vocab_size).to(self.device)

            value = self._reward(state, prob)
            return value

        # Select next token
        token = self._select(s)

        # Recursevily call step until a leaf node is found
        next_state = self._get_next_state(state, token)

        #print("\t selected:", token)
        value = self.step(next_state, prob + torch.log(self.Ps[s][token]))

        self.Qsa[s][token] = (self.Nsa[s][token] * self.Qsa[s][token] + value) / (self.Nsa[s][token] + 1)
        self.Nsa[s][token] += 1
        self.Ns[s] += 1

        return value

    def _expand(self, state):
        with torch.no_grad():
            #print("\t expand:", state)
            y_i = self.language_model(state)[:,-1,:]
            y_i = filter_top_k(y_i, self.k)
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

    def _select(self, state, eps=1e-8):
        puct = self.Qsa[state] + self.c * self.Ps[state] * np.sqrt(self.Ns[state] + eps)/(
                    1 + self.Nsa[state])

        return torch.argmax(puct)
