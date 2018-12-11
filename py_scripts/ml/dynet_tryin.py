import random
import matplotlib.pyplot as plt
import numpy as np
# import dynet_config
# dynet_config.set_gpu()

import dynet as dy

from ml_utils import init_model, read_from_stdin, read_training_data_from_file

HIDDEN_NUM = 2

training_data = list(filter(None, read_training_data_from_file('tmp.txt')))
random.shuffle(training_data)
# training_data = training_data[:600]

training_subset_size = 400

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
    loss = dy.scalarInput(0)
    batch_loss = []
    epoch_losses = []

    input_dim = len(training_data[0][0][0])
    output_dim = len(training_data[0][0][1])
    model_input_l, hidden_l, model_output_l, model = init_model(input_dim, HIDDEN_NUM, output_dim)
    trainer = dy.MomentumSGDTrainer(model)

    # global big_loss, batch_loss, letter, loss
    for i in range(num_of_epochs):
        random.shuffle(training_data)
        epoch_losses = []
        training_data_subset = training_data[:training_subset_size]
        for word in training_data_subset:
            batch_loss = []
            for letter in word:
                model_input_l.set(letter[0])
                target = letter[1].index(1)

                loss = -dy.log(dy.pick(dy.softmax(model_output_l), target))
                batch_loss.append(loss)
                loss.backward()
                trainer.update()

            epoch_losses.extend(batch_loss)
        print("epoch " + str(i))
        print("learning_rate:", trainer.learning_rate)
        print("epoch avg loss:", (dy.esum(epoch_losses) / len(epoch_losses)).npvalue()[0])
        print("epoch max loss:", dy.emax(epoch_losses).npvalue()[0])
        print("===")
        # trainer.learning_rate = trainer.learning_rate*0.8

    names_0 = []
    names_1 = []
    values_0 = []
    values_1 = []

    hidden_0_dict = {}
    hidden_1_dict = {}

    for letter in alphabet:
        model_input_l.set([int(i) for i in onehot_encode_char(alphabet, letter)])
        hidden_0_dict[letter] = hidden_l.npvalue()[0]
        hidden_1_dict[letter] = hidden_l.npvalue()[1]

    for name in alphabet:
        if name in consonants:
            names_0.append(name)
            names_1.append(name)
            values_0.append(hidden_0_dict[name])
            values_1.append(hidden_1_dict[name])

    for name in alphabet:
        if name in vovels:
            names_0.append(name)
            names_1.append(name)
            values_0.append(hidden_0_dict[name])
            values_1.append(hidden_1_dict[name])

    if (np.mean(values_0[-len(vovels):]) < np.mean(values_1[-len(vovels):])):
        plt.plot(names_0, values_0, "C0+")
        plt.plot(names_1, values_1, "C1o")
    else:
        plt.plot(names_0, values_0, "C1o")
        plt.plot(names_1, values_1, "C0+")


plt.ylim(0, 1)
initial_num_of_epochs = 10
for i in range(10):
    num_of_epochs = (2 ** (i + 2))
    train_ml(num_of_epochs)
    plt.savefig('plots/plot' + str(i) + '.png')
