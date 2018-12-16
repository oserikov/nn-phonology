import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# import dynet_config
# dynet_config.set_gpu()

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

training_data = list(filter(None, read_training_data_from_file('tmp.txt')))
random.shuffle(training_data)
# training_data = training_data[:600]

training_subset_size = 600

alphabet = []
with open(r"data/tur_alphabet_wiki.txt", encoding='utf-8') as f:
    alphabet = [l.strip() for l in f]
vovels = [c for c in "aeiıoöüu"]
consonants = set(alphabet).difference(set(vovels))


def onehot_encode_char(alphabet, char):
    onehot_encoding_char = ["0"] * len(alphabet)
    onehot_encoding_char[alphabet.index(char)] = "1"
    return onehot_encoding_char


def train_ml(num_of_epochs=10):
    # loss = dy.scalarInput(0)
    # batch_loss = []
    # epoch_losses = []

    input_dim = len(training_data[0][0][0])
    output_dim = len(training_data[0][0][1])
    model_input_l, hidden_l, model_output_l, model = init_model(input_dim, HIDDEN_NUM, output_dim)
    trainer = dy.MomentumSGDTrainer(model)

    for i in range(num_of_epochs):
        random.shuffle(training_data)
        epoch_losses = []
        training_data_subset = training_data[:training_subset_size]
        val_data_subset = training_data[training_subset_size:]
        for word in training_data_subset:
            for letter in word:
                model_input_l.set(letter[0])
                target = letter[1].index(1)

                loss = -dy.log(dy.pick(dy.softmax(model_output_l), target))
                loss.backward()
                trainer.update()

        for word in val_data_subset:
            batch_loss = []
            for letter in word:
                model_input_l.set(letter[0])
                target = letter[1].index(1)

                loss = -dy.log(dy.pick(dy.softmax(model_output_l), target))
                batch_loss.append(loss)
            epoch_losses.extend(batch_loss)
        print("epoch " + str(i))
        print("learning_rate:", trainer.learning_rate)
        print("epoch avg loss:", (dy.esum(epoch_losses) / len(epoch_losses)).npvalue()[0])
        print("epoch max loss:", dy.emax(epoch_losses).npvalue()[0])
        print("===")
        # trainer.learning_rate = trainer.learning_rate*0.8

    names = []
    values = [[] for i in range(HIDDEN_NUM)]

    hidden_dict = [{} for i in range(HIDDEN_NUM)]

    for letter in alphabet:
        model_input_l.set([int(bit) for bit in onehot_encode_char(alphabet, letter)])
        for i in range(HIDDEN_NUM):
            hidden_dict[i][letter] = hidden_l.npvalue()[i]

    for name in consonants:
        names.append(name)
        for i in range(HIDDEN_NUM):
            values[i].append(hidden_dict[i][name])

    for name in vovels:
        names.append(name)
        for i in range(HIDDEN_NUM):
            values[i].append(hidden_dict[i][name])

    for row in range(HIDDEN_NUM):
        plt.ylim(0, 1)
        plt.plot(names, values[row], "o")
        plt.savefig(PLOTS_DIRNAME + '/unit_' + str(row) + '_e' + str(num_of_epochs) + '.png')
        plt.clf()


initial_num_of_epochs = 10
for i in range(10):
    num_of_epochs = (10 * (i + 1))
    train_ml(num_of_epochs)
