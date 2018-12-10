import random
import matplotlib.pyplot as plt

import dynet_config
dynet_config.set(mem=4096)

import dynet as dy

from ml_utils import init_model, read_from_stdin, read_training_data_from_file

HIDDEN_NUM = 2


training_data = read_training_data_from_file('tmp.txt')
num_of_epochs = 20
training_subset_size = 1000

input_dim = len(training_data[0][0][0])
output_dim = len(training_data[0][0][1])
model_input_l, hidden_l, model_output_l, model = init_model(input_dim, HIDDEN_NUM, output_dim)
trainer = dy.SimpleSGDTrainer(model)

loss = dy.scalarInput(0)
batch_loss = []
big_loss = []
for i in range(num_of_epochs):
    random.shuffle(training_data)
    for word in list(filter(None, training_data[:training_subset_size])):
        batch_loss = []
        for letter in word:
            model_input_l.set(letter[0])
            target = letter[1].index(1)

            loss = dy.pickneglogsoftmax(model_output_l, target)
            batch_loss.append(loss)
            loss.backward()

        # dy.esum(batch_loss).backward()
        trainer.update()

        big_loss.extend(batch_loss)
    print((dy.esum(batch_loss) / len(batch_loss)).npvalue())


def onehot_encode_char(alphabet, char):
    onehot_encoding_char = ["0"] * len(alphabet)
    onehot_encoding_char[alphabet.index(char)] = "1"
    return onehot_encoding_char


names_0 = []
names_1 = []
values_0 = []
values_1 = []

hidden_0_dict = {}
hidden_1_dict = {}



alphabet = []
with open(r"data/tur_alphabet_wiki.txt", encoding='utf-8') as f:
    alphabet = [l.strip() for l in f]
vovels = [c for c in "aeiıoöüu"]
consonants = set(alphabet).difference(set(vovels))

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

# print(names_0, names_1, values_0, values_1)

plt.plot(names_0, values_0, "C0", names_1, values_1, "C1")
plt.savefig('plot.png')
