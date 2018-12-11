import random
import matplotlib.pyplot as plt

# import dynet_config
# dynet_config.set_gpu()

import dynet as dy

from ml_utils import init_model, read_from_stdin, read_training_data_from_file

HIDDEN_NUM = 2


training_data = read_training_data_from_file('tmp.txt')
random.shuffle(training_data)
# training_data = training_data[:600]
num_of_epochs = 60
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

def train_ml():
    loss = dy.scalarInput(0)
    batch_loss = []
    big_loss = []

    input_dim = len(training_data[0][0][0])
    output_dim = len(training_data[0][0][1])
    model_input_l, hidden_l, model_output_l, model = init_model(input_dim, HIDDEN_NUM, output_dim)
    trainer = dy.AdamTrainer(model)#, learning_rate_max=0.001, learning_rate_min=0.00001)

    # global big_loss, batch_loss, letter, loss
    for i in range(num_of_epochs):
        random.shuffle(training_data)
        big_loss = []
        for word in list(filter(None, training_data[:training_subset_size])):
            batch_loss = []
            last_letter = [0] * input_dim
            for letter in word:
                for i in range(input_dim):
                    if last_letter[i] != letter[0][i] and last_letter.count(0) != input_dim:
                        print(last_letter)
                        print(letter[0])
                        print()
                        break

                model_input_l.set(letter[0])
                target = letter[1].index(1)

                loss = dy.pickneglogsoftmax(model_output_l, target)
                batch_loss.append(loss)
                last_letter = letter[1]
                loss.backward()

            # dy.esum(batch_loss).backward()
            trainer.update()
            big_loss.extend(batch_loss)

        # trainer.learning_rate *= 0.8
        print(trainer.learning_rate, (dy.esum(big_loss) / len(big_loss)).npvalue())
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

    plt.plot(names_0, values_0, "C0")
    plt.plot(names_1, values_1, "C1")


for i in range(10):
    train_ml()
    plt.savefig('plots/plot' + str(i) + '.png')


