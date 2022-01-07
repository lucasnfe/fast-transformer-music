#
# Generate MIDI piano pieces with fast transformer.
#
# Author: Lucas N. Ferreira - lucasnfe@gmail.com
#
#

import torch
import math
import argparse
import encoder

from torch.distributions.categorical import Categorical
from models.music_generator_recurrent import RecurrentMusicGenerator

START_TOKEN = 388

def filter_top_p(y_hat, p, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(y_hat, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    y_hat = y_hat.masked_fill(indices_to_remove, filter_value)

    return y_hat

def filter_top_k(y_hat, k, filter_value=-float("Inf")):
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = y_hat < torch.topk(y_hat, k)[0][..., -1, None]
    y_hat = y_hat.masked_fill(indices_to_remove, filter_value)

    return y_hat

def filter_note_off(y_hat, notes_status, filter_value=-float("Inf")):
    for i in range(len(notes_status)):
        if notes_status[i] == 0:
            y_hat[:,encoder.START_IDX['note_off'] + i] = filter_value

    return y_hat

def filter_velocity(y_hat, status_velocity, filter_value=-float("Inf")):
    if status_velocity == 1:
        i = encoder.RANGE_VEL
        y_hat[:,encoder.START_IDX['velocity']:encoder.START_IDX['velocity']+i] = filter_value

    return y_hat

def filter_time_shift(y_hat, status_time_shift, filter_value=-float("Inf")):
    if status_time_shift == 1:
        i = encoder.RANGE_TIME_SHIFT
        y_hat[:,encoder.START_IDX['time_shift']:encoder.START_IDX['time_shift']+i] = filter_value

    return y_hat

def sample_tokens(y_hat, num_samples=1):
    # Sample from filtered categorical distribution
    probs = torch.softmax(y_hat, dim=1)
    random_idx = torch.multinomial(probs, num_samples)
    return random_idx

def generate(model, prime, seq_len, k=0, p=0, temperature=1.0):
    memory = None
    piece = []

    # Process prime sequence
    prime_len = prime.shape[1]
    for i in range(prime_len):
        x_i = prime[:,i:i+1]
        y_i, memory = model(x_i, i=i, memory=memory)
        piece.append(int(x_i))

    # Generate new tokens
    for i in range(prime_len, seq_len):
        status_notes, status_velocity, status_time_shift = get_piece_status(piece)

        # y_i = filter_note_off(y_i, status_notes)
        # y_i = filter_velocity(y_i, status_velocity)
        # y_i = filter_time_shift(y_i, status_time_shift)

        # Apply temperature filter
        y_i = y_i/temperature

        # Apply top_k and top_p filters
        if k > 0:
            y_i = filter_top_k(y_i, k)
        if p > 0 and p < 1.0:
            y_i = filter_top_p(y_i, p)

        # sample new token
        x_i = sample_tokens(y_i)

        # Stop if end-of-piece token
        event = encoder.Event.from_int(int(x_i))
        if event.type == 'special' and event.value == 1:
            print("End of piece reached...", "stopping...")
            break

        y_i, memory = model(x_i, i=i, memory=memory)
        piece.append(int(x_i))

    return piece

def get_piece_status(piece):
    status_notes = [0 for i in range(128)]

    for token in piece:
        status_velocity = 0
        status_time_shift = 0

        event = encoder.Event.from_int(token)

        if event.type == 'note_on':
            status_notes[event.value] = 1
        elif event.type == 'note_off':
            status_notes[event.value] = 0
        elif event.type == 'velocity':
            status_velocity = 1
        elif event.type == 'time_shift':
            if event.value != encoder.RANGE_TIME_SHIFT - 1:
                status_time_shift = 1

    return status_notes, status_velocity, status_time_shift

def generate_beam_search(model, prime, n, beam_size, k=0, p=0, temperature=1.0):
    # Process prime sequence
    memory = None
    y_hat = []
    x_hat = []

    # batch size and number of words
    batch_size = prime.shape[0]

    # expand to beam size the source latent representations / source lengths
    prime = prime.unsqueeze(1).expand((batch_size, beam_size) + prime.shape[1:]).contiguous().view((batch_size * beam_size,) + prime.shape[1:])

    prime_len = prime.shape[1]
    for i in range(prime_len):
        x_hat.append(prime[:,i:i+1])
        y_i, memory = model(x_hat[-1], i=i, memory=memory)
        y_hat.append(y_i)

    # Get vocabulary size from logits
    vocab_size = y_hat[-1].shape[-1]

    # Create first beam
    scores = torch.nn.functional.log_softmax(y_hat[-1], dim=-1)

    # Apply temperature, top_k and top_p filters
    scores = scores/temperature

    if k > 0:
        scores = filter_top_k(scores, k)
    if p > 0 and p < 1.0:
        scores = filter_top_p(scores, p)

    first_words = sample_tokens(scores, num_samples=beam_size)[0]
    first_scores = scores[0][first_words]

    first_scores, _indices = torch.sort(first_scores, descending=True, dim=0)
    first_words = first_words[_indices]

    beam_scores = first_scores
    beam_candidates = torch.cat((prime, first_words.view(beam_size, batch_size)), dim=1)

    x_hat.append(first_words.view(-1))

    # current position
    for i in range(prime_len, prime_len + n):
        # Compute scores
        y_i, memory = model(x_hat[-1], i=i, memory=memory)
        y_hat.append(y_i)

        scores = torch.nn.functional.log_softmax(y_hat[-1], dim=-1)
        _scores = scores + beam_scores[:, None].expand_as(scores)

        # Apply temperature, top_k and top_p filters
        _scores = _scores/temperature

        if k > 0:
            _scores = filter_top_k(_scores, k)
        if p > 0 and p < 1.0:
            _scores = filter_top_p(_scores, p)

        _scores = _scores.view(batch_size, beam_size * vocab_size)

        next_words = sample_tokens(_scores, num_samples=beam_size)
        next_scores = _scores.gather(1, next_words)

        next_scores, _indices = torch.sort(next_scores, descending=True, dim=1)
        next_words = next_words.gather(1, _indices)

        beam_ids = next_words // vocab_size
        word_ids = next_words % vocab_size

        # Generate new beam candidates
        beam_candidates = beam_candidates[beam_ids].squeeze()
        beam_candidates = torch.cat((beam_candidates, word_ids.view(beam_size, batch_size)), dim=1)
        beam_scores = next_scores.view(-1)

        print(beam_candidates)
        print(beam_scores)

        x_hat.append(word_ids.view(-1))

    # Get best candidate
    best_id = torch.argmax(beam_scores)
    best_candidate = beam_candidates[best_id]

    print("best_candidate", best_candidate)

    return [int(token) for token in best_candidate]

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='generate.py')
    parser.add_argument('--model', type=str, required=True, help="Path to load model from.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--k', type=int, default=0, help="Number k of elements to consider while sampling.")
    parser.add_argument('--p', type=float, default=1.0, help="Probability p to consider while sampling.")
    parser.add_argument('--t', type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument('--vocab_size', type=int, required=True, help="Vocabulary size.")
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

    # Build linear transformer
    model = RecurrentMusicGenerator(n_tokens=opt.vocab_size,
                                     d_query=opt.d_query,
                                     d_model=opt.d_query * opt.n_heads,
                                     seq_len=opt.seq_len,
                              attention_type="linear",
                                    n_layers=opt.n_layers,
                                     n_heads=opt.n_heads).to(device)

    # Load model
    model.load_state_dict(torch.load(opt.model, map_location=device)["model_state"])
    model.eval()

    # Define prime sequence
    with torch.no_grad():
        prime = [START_TOKEN]
        prime = torch.tensor(prime).unsqueeze(dim=0).to(device)

        # Generate continuation
        # piece = generate_beam_search(model, prime, n=1000, beam_size=8, k=opt.k, p=opt.p, temperature=opt.t)
        piece = generate(model, prime, opt.seq_len, opt.k, opt.p, temperature=opt.t)

    encoder.decode_midi(piece, opt.save_to)
    print(piece)
