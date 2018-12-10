import dynet as dy
import random
import matplotlib.pyplot as plt
import numpy as np
import datetime

from ml_utils import init_model, read_from_stdin

HIDDEN_NUM = 2

alphabet = []
with open(r"data/tur_alphabet_wiki.txt", encoding='utf-8') as f:
    alphabet = [l.strip() for l in f]

vovels = [c for c in "aeiıoöüu"]
consonants = set(alphabet).difference(set(vovels))


def train_model(HIDDEN_NUM, training_data, num_of_epochs=10, training_subset_size=500):
    input_dim = len(training_data[0][0][0])
    output_dim = len(training_data[0][0][1])
    model_input_l, hidden_l, model_output_l, model = init_model(input_dim, HIDDEN_NUM, output_dim)
    trainer = dy.MomentumSGDTrainer(model)

    batch_loss = []
    big_loss = []
    for i in range(num_of_epochs):
        random.shuffle(training_data)
        for word in training_data[:training_subset_size]:
            batch_loss = []
            for letter in word:
                model_input_l.set(letter[0])
                target = letter[1].index(1)

                loss = dy.pickneglogsoftmax(model_output_l, target)
                batch_loss.append(loss)

            dy.esum(batch_loss).backward()
            trainer.update()

            big_loss.extend(batch_loss)
        print((dy.esum(batch_loss) / len(batch_loss)).npvalue())

    return model_input_l, hidden_l, model_output_l


def onehot_encode_char(alphabet, char):
    onehot_encoding_char = ["0"] * len(alphabet)
    onehot_encoding_char[alphabet.index(char)] = "1"
    return onehot_encoding_char


num_of_iters = 1
names_0 = [[]]
names_1 = [[]]
values_0 = [[]]
values_1 = [[]]
for i in range(num_of_iters):
    model_input_l, hidden_l, model_output_l = train_model(HIDDEN_NUM, read_from_stdin())

    hidden_0_dict = {}
    hidden_1_dict = {}

    for letter in alphabet:
        model_input_l.set([int(i) for i in onehot_encode_char(alphabet, letter)])
        hidden_0_dict[letter] = hidden_l.npvalue()[0]
        hidden_1_dict[letter] = hidden_l.npvalue()[1]

    for name in alphabet:
        if name in consonants:
            names_0[i].append(name)
            names_1[i].append(name)
            values_0[i].append(hidden_0_dict[name])
            values_1[i].append(hidden_1_dict[name])

    for name in alphabet:
        if name in vovels:
            names_0[i].append(name)
            names_1[i].append(name)
            values_0[i].append(hidden_0_dict[name])
            values_1[i].append(hidden_1_dict[name])

    plt.plot(names_0[i], values_0[i], "C0")
    plt.plot(names_1[i], values_1[i], "C1")
    plt.savefig(str(datetime.datetime.now().timestamp()) + '_hl_' + str(HIDDEN_NUM) + '_' + str(i) + '.png')
