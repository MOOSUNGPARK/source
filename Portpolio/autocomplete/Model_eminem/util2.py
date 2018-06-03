# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import glob
import sys

# size of the alphabet that we work with
ALPHASIZE = 98


# Specification of the supported alphabet (subset of ASCII-7)
# 10 line feed LF
# 32-64 numbers and punctuation
# 65-90 upper-case letters
# 91-97 more punctuation
# 97-122 lower-case letters
# 123-126 more punctuation

def encode_text(vocab, txt_list):
    return list(map(lambda a: vocab[a], txt_list))

def decode_text(vocab, txt_list):
    return list(map(lambda a: vocab[a] if vocab[a] != '\n' else '\\', txt_list))

def sample_from_probabilities(vocab_size, probabilities, topn=10):
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(vocab_size, 1, p=p)[0]

def txt_one_hot(data, vocab_size):
    _data = np.zeros((len(data), vocab_size))

    for idx, string in enumerate(data):
        _data[idx, string] = 1.0

    return _data

def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs, vocab_size):
    data = np.array(raw_data)


    data = txt_one_hot(data, vocab_size)


    data_len = data.shape[0]
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    # xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    # ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size, vocab_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size, vocab_size])



    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size, :]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size, :]
            x = np.roll(x, -epoch, axis=0)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch


def find_book(index, bookranges):
    return next(
        book["name"] for book in bookranges if (book["start"] <= index < book["end"]))


def find_book_index(index, bookranges):
    return next(
        i for i, book in enumerate(bookranges) if (book["start"] <= index < book["end"]))


def print_learning_learned_comparison(Vocab, X, Y, batch_size, seq_len, losses, bookranges, batch_loss, batch_accuracy, epoch_size, index, epoch):
    """Display utility for printing learning statistics"""
    print()
    # epoch_size in number of batches
    # batch_size = X.shape[0]  # batch_size in number of sequences
    # sequence_len = X.shape[1]  # sequence_len in number of characters

    start_index_in_epoch = index % (epoch_size * batch_size * seq_len)
    for k in range(batch_size):
        index_in_epoch = index % (epoch_size * batch_size * seq_len)
        decx = ''.join(decode_text(Vocab, X[k]))
        decy = ''.join(decode_text(Vocab, Y[k]))
        bookname = find_book(index_in_epoch, bookranges)
        formatted_bookname = "{: <10.40}".format(bookname)  # min 10 and max 40 chars
        epoch_string = "{:4d}".format(index) + " (epoch {}) ".format(epoch)
        loss_string = "loss: {:.5f}".format(losses[k])
        print_string = epoch_string + formatted_bookname + " │ {} │ {} │ {}"
        print(print_string.format(decx, decy, loss_string))
        index += seq_len
    # box formatting characters:
    # │ \u2502
    # ─ \u2500
    # └ \u2514
    # ┘ \u2518
    # ┴ \u2534
    # ┌ \u250C
    # ┐ \u2510
    format_string = "└{:─^" + str(len(epoch_string)) + "}"
    format_string += "{:─^" + str(len(formatted_bookname)) + "}"
    # format_string += "┴{:─^" + str(len(decx) + 2) + "}"
    # format_string += "┴{:─^" + str(len(decy) + 2) + "}"
    format_string += "┴{:─^" + str(len(loss_string)) + "}┘"
    # footer = format_string.format('INDEX', 'BOOK NAME', 'LOSS')
    # print(footer)
    # print statistics
    batch_index = start_index_in_epoch // (batch_size * seq_len)
    batch_string = "batch {}/{} in epoch {},".format(batch_index, epoch_size, epoch)
    stats = "{: <28} batch loss: {:.5f}, batch accuracy: {:.5f}".format(batch_string, batch_loss, batch_accuracy)
    print()
    print("TRAINING STATS: {}".format(stats))


class Progress:
    """Text mode progress bar.
    Usage:
            p = Progress(30)
            p.step()
            p.step()
            p.step(start=True) # to restart form 0%
    The progress bar displays a new header at each restart."""
    def __init__(self, maxi, size=100, msg=""):
        """
        :param maxi: the number of steps required to reach 100%
        :param size: the number of characters taken on the screen by the progress bar
        :param msg: the message displayed in the header of the progress bat
        """
        self.maxi = maxi
        self.p = self.__start_progress(maxi)()  # () to get the iterator from the generator
        self.header_printed = False
        self.msg = msg
        self.size = size

    def step(self, reset=False):
        if reset:
            self.__init__(self.maxi, self.size, self.msg)
        if not self.header_printed:
            self.__print_header()
        next(self.p)

    def __print_header(self):
        print()
        format_string = "0%{: ^" + str(self.size - 6) + "}100%"
        print(format_string.format(self.msg))
        self.header_printed = True

    def __start_progress(self, maxi):
        def print_progress():
            # Bresenham's algorithm. Yields the number of dots printed.
            # This will always print 100 dots in max invocations.
            dx = maxi
            dy = self.size
            d = dy - dx
            for x in range(maxi):
                k = 0
                while d >= 0:
                    print('=', end="", flush=True)
                    k += 1
                    d -= dx
                d += dy
                yield k

        return print_progress

def read_data_files(directory, validation=True):
    """Read data files according to the specified glob pattern
    Optionnaly set aside the last file as validation data.
    No validation data is returned if there are 5 files or less.
    :param directory: for example "data/*.txt"
    :param validation: if True (default), sets the last file aside as validation data
    :return: training data, validation data, list of loaded file names with ranges
     If validation is
    """
    codetext = []
    bookranges = []
    shakelist = glob.glob(directory, recursive=True)
    for shakefile in shakelist:
        try :
            shaketext = open(shakefile, "r")
            start = len(codetext)
            codetext.extend(shaketext.read())

        except :
            shaketext = open(shakefile, "r", encoding='UTF8')
            start = len(codetext)
            codetext.extend(shaketext.read())

        print("Loading file " + shakefile)
        end = len(codetext)
        bookranges.append({"start": start, "end": end, "name": shakefile.rsplit("/", 1)[-1]})
        shaketext.close()

    if len(bookranges) == 0:
        sys.exit("No training data has been found. Aborting.")

    # For validation, use roughly 90K of text,
    # but no more than 10% of the entire text
    # and no more than 1 book in 5 => no validation at all for 5 files or fewer.

    # 10% of the text is how many files ?

    vocab = sorted(list(set(codetext)))
    c_to_i_vocab = dict((c,i) for i,c in enumerate(vocab))
    i_to_c_vocab = dict((i,c) for i,c in enumerate(vocab))
    codetext = encode_text(c_to_i_vocab, codetext)
    total_len = len(codetext)
    validation_len = 0
    nb_books1 = 0
    for book in reversed(bookranges):
        validation_len += book["end"]-book["start"]
        nb_books1 += 1
        if validation_len > total_len // 10:
            break

    # 90K of text is how many books ?
    validation_len = 0
    nb_books2 = 0
    for book in reversed(bookranges):
        validation_len += book["end"]-book["start"]
        nb_books2 += 1
        if validation_len > 90*1024:
            break

    # 20% of the books is how many books ?
    nb_books3 = len(bookranges) // 5

    # pick the smallest
    nb_books = min(nb_books1, nb_books2, nb_books3)

    if nb_books == 0 or not validation:
        cutoff = len(codetext)
    else:
        cutoff = bookranges[-nb_books]["start"]
    valitext = codetext[cutoff:]
    codetext = codetext[:cutoff]
    return c_to_i_vocab, i_to_c_vocab, codetext, valitext, bookranges


def print_data_stats(datalen, valilen, epoch_size):
    datalen_mb = datalen/1024.0/1024.0
    valilen_kb = valilen/1024.0
    print("Training text size is {:.2f}MB with {:.2f}KB set aside for validation.".format(datalen_mb, valilen_kb)
          + " There will be {} batches per epoch".format(epoch_size))


def print_validation_header(validation_start, bookranges):
    bookindex = find_book_index(validation_start, bookranges)
    books = ''
    for i in range(bookindex, len(bookranges)):
        books += bookranges[i]["name"]
        if i < len(bookranges)-1:
            books += ", "
    print("{: <60}".format("Validating on " + books), flush=True)


def print_validation_stats(loss, accuracy):
    print("VALIDATION STATS:                                  loss: {:.5f},       accuracy: {:.5f}".format(loss,
                                                                                                           accuracy))


def print_text_generation_header():
    print()
    print("┌{:─^111}┐".format('Generating random text from learned state'))


def print_text_generation_footer():
    print()
    print("└{:─^111}┘".format('End of generation'))


def frequency_limiter(n, multiple=1, modulo=0):
    def limit(i):
        return i % (multiple * n) == modulo*multiple
    return limit