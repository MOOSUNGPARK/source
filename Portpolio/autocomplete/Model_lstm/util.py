import numpy as np
import random

def txt_one_hot(data, vocab):
    _data = np.zeros((len(data), len(vocab)))

    for idx, string in enumerate(data):
        _data[idx, vocab[string]] = 1.0

    return _data

def load_txt_vocab(loc):
    try :
        txt = ''
        with open(loc, 'r') as f:
            txt += f.read()
    except:
        txt = ''
        with open(loc, 'r', encoding='UTF8') as f:
            txt += f.read()
    txt = txt.lower()
    vocab = sorted(list(set(txt)))
    c_to_i_vocab = dict((c,i) for i,c in enumerate(vocab))
    i_to_c_vocab = dict((i,c) for i,c in enumerate(vocab))
    return txt, c_to_i_vocab, i_to_c_vocab

def make_batch(data, batch_range, shape):
    batch_size, time_steps, size = shape
    x_batch = np.zeros((batch_size, time_steps, size))
    y_batch = np.zeros((batch_size, time_steps, size))

    chosen_idx_list = random.sample(batch_range, batch_size)

    for step in range(time_steps):
        x_idx = [idx + step for idx in chosen_idx_list]
        y_idx = [idx + step + 1 for idx in chosen_idx_list]

        x_batch[:,step,:] = data[x_idx, :]
        y_batch[:,step,:] = data[y_idx, :]

    return x_batch, y_batch

def make_batch2(data, batch_idx_list, shape):
    batch_size, time_steps, size = shape
    x_batch = np.zeros((batch_size, time_steps, size))
    y_batch = np.zeros((batch_size, time_steps, size))

    for step in range(time_steps):
        x_idx = [idx + step for idx in batch_idx_list]
        y_idx = [idx + step + 1 for idx in batch_idx_list]

        x_batch[:,step,:] = data[x_idx, :]
        y_batch[:,step,:] = data[y_idx, :]

    return x_batch, y_batch

