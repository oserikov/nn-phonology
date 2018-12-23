# -*- coding: utf-8 -*-

import random
import sys
import os
import matplotlib
from matplotlib import pyplot as plt

from py_scripts.ml.ml_utils import read_training_data_from_file, read_from_stdin
from py_scripts.ml.model_dynet import ModelDyNet
from py_scripts.ml.model_pytorch import ModelPyTorch


def initialize_hidden_num():
    hidden_num = 2
    if len(sys.argv) > 1:
        # noinspection PyBroadException
        try:
            hidden_num = int(sys.argv[1])
        except:
            pass

    plots_dir_name = os.path.join("plots", str(hidden_num) + "_hidden_l")

    if not os.path.exists(plots_dir_name):
        os.makedirs(plots_dir_name)

    return hidden_num, plots_dir_name


HIDDEN_NUM, PLOTS_DIR_NAME = initialize_hidden_num()

print("HIDDEN LAYERS NUM: " + str(HIDDEN_NUM))

training_data = list(filter(None, read_from_stdin()))
# training_data = list(filter(None, read_training_data_from_file('tmp.txt')))
random.shuffle(training_data)

training_subset_size = 1100


def onehot_encode_char(alphabet, char):
    onehot_encoding_char = ["0"] * len(alphabet)
    onehot_encoding_char[alphabet.index(char)] = "1"
    return onehot_encoding_char


def train_ml(num_of_epochs=10):
    input_dim = len(training_data[0][0][0])
    output_dim = len(training_data[0][0][1])

    model = ModelPyTorch(input_dim, HIDDEN_NUM, output_dim, learning_rate=0.0001)

    for i in range(num_of_epochs):
        random.shuffle(training_data)

        training_data_subset = training_data[:training_subset_size]
        val_data_subset = training_data[training_subset_size:]

        _, epoch_loss = model.train_batches(training_data_subset)
        # _, epoch_loss = model.predict_batches(val_data_subset)
        print("epoch " + str(i) + ", epoch avg loss:", epoch_loss)

    return model


def plot_ml(model):
    ORDERED_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
    alphabet, vowels = get_alphabet_and_vowels()
    hidden_dict = get_units_activation_levels(alphabet, model)

    names = []
    values = [[] for _ in range(HIDDEN_NUM)]

    for name in ORDERED_ALPHABET:
        names.append(r'$\mathrm{' + name + '}$')
        for i in range(HIDDEN_NUM):
            values[i].append(hidden_dict[i][name])

    matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath},\usepackage{amssymb}')
    for row in range(HIDDEN_NUM):
        ax = plt.subplot(111)

        ax.set_ylim([-0.1, 1.05])
        for i, name, value in zip(range(len(names)), names, values[row]):
            if name[-3] in vowels:
                ax.plot(i, value, "o", mfc='none', color='C0')
            else:
                ax.plot(i, value, "o", color='C0')
            ax.annotate(name, xy=(i - 0.3, value - 0.06))

        ax.set_yticks((0, 1))
        ax.set_xticks(())

        plt.savefig(PLOTS_DIR_NAME + '/unit_' + str(row) + '.png')

        plt.clf()


def get_alphabet_and_vowels():
    alphabet = []
    with open(r"data/tur_alphabet_wiki.txt", encoding='utf-8') as f:
        alphabet = [l.strip() for l in f]
    # noinspection SpellCheckingInspection
    vowels = [c for c in "aeiıoöüu"]
    return alphabet, vowels


def get_units_activation_levels(alphabet, model):
    hidden_dict = [{} for _ in range(HIDDEN_NUM)]
    for letter in alphabet:
        onehot_encoded_letter = [int(bit) for bit in onehot_encode_char(alphabet, letter)]
        model.predict_batches([[[onehot_encoded_letter, [1] * len(onehot_encoded_letter)]]])
        for i in range(HIDDEN_NUM):
            hidden_dict[i][letter] = model.get_context_state()[i]
    return hidden_dict


plot_ml(train_ml(10000))
