# -*- coding: utf-8 -*-

import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import dynet as dy

from ml_utils import init_model, read_from_stdin, read_training_data_from_file

HIDDEN_NUM = 2
if len(sys.argv) > 1:
    try:
        HIDDEN_NUM = int(sys.argv[1])
    except:
        pass

PLOTS_DIRNAME = os.path.join("plots", str(HIDDEN_NUM) + "_hidden_l")

if not os.path.exists(PLOTS_DIRNAME):
    os.makedirs(PLOTS_DIRNAME)

print("HIDDEN LAYERS NUM: " + str(HIDDEN_NUM))

training_data = list(filter(None, read_from_stdin()))  #list(filter(None, read_training_data_from_file('tmp.txt')))
random.shuffle(training_data)
# training_data = training_data[:600]

training_subset_size = 1100

print(len(training_data))

alphabet = []
with open(r"data/tur_alphabet_wiki.txt", encoding='utf-8') as f:
    alphabet = [l.strip() for l in f]
vovels = [c for c in "aeiıoöüu"]
consonants = sorted(set(alphabet).difference(set(vovels)))


def onehot_encode_char(alphabet, char):
    onehot_encoding_char = ["0"] * len(alphabet)
    onehot_encoding_char[alphabet.index(char)] = "1"
    return onehot_encoding_char


def train_ml(num_of_epochs=10):
    input_dim = len(training_data[0][0][0])
    output_dim = len(training_data[0][0][1])
    layers = 1

    model = dy.ParameterCollection()

    rnn = dy.SimpleRNNBuilder(layers, input_dim, HIDDEN_NUM, model)
    R = model.add_parameters((output_dim, HIDDEN_NUM))

    trainer = dy.MomentumSGDTrainer(model)

    for i in range(num_of_epochs):
        random.shuffle(training_data)
        epoch_losses = []

        training_data_subset = training_data[:training_subset_size]
        val_data_subset = training_data[training_subset_size:]

        for word in training_data_subset:
            dy.renew_cg()
            s = rnn.initial_state()

            batch_loss = []

            for letter in word:
                s = s.add_input(dy.inputVector(letter[0]))
                target = letter[1].index(1)

                loss = dy.pickneglogsoftmax(R * s.output(), target)
                batch_loss.append(loss)

            batch_loss = dy.esum(batch_loss)
            batch_loss.backward()
            trainer.update()

        for word in val_data_subset:
            dy.renew_cg()
            s = rnn.initial_state()

            batch_loss = []
            for letter in word:
                s = s.add_input(dy.inputVector(letter[0]))
                target = letter[1].index(1)

                loss = dy.pickneglogsoftmax(R * s.output(), target)
                batch_loss.append(loss.npvalue())

            epoch_losses.extend(batch_loss)

        print("epoch " + str(i) + ", epoch avg loss:", np.mean(epoch_losses))

    names = []
    values = [[] for _ in range(HIDDEN_NUM)]

    hidden_dict = [{} for _ in range(HIDDEN_NUM)]

    for letter in alphabet:
        dy.renew_cg()
        s = rnn.initial_state()

        s = s.add_input(dy.inputVector(
            [int(bit) for bit in onehot_encode_char(alphabet, letter)])
        )

        output = s.output()
        for i in range(HIDDEN_NUM):
            hidden_dict[i][letter] = output.npvalue()[i]

    for name in consonants:
        names.append(r'$\mathit{'+name+'}$')
        for i in range(HIDDEN_NUM):
            values[i].append(hidden_dict[i][name])

    for name in vovels:
        names.append(r'$\mathit{' + name + '}$')
        for i in range(HIDDEN_NUM):
            values[i].append(hidden_dict[i][name])

    print(names)

    for row in range(HIDDEN_NUM):
        plt.ylim(-1, 1)

        plt.plot(names, values[row], "o")

        plt.savefig(PLOTS_DIRNAME + '/unit_' + str(row) + '_e' + str(num_of_epochs) + '.png')

        plt.clf()


matplotlib.rc('font', **{'sans-serif' : 'Arial',
                         'family' : 'sans-serif'})

train_ml(300)
