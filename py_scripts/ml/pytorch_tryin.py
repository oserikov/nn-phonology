import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import random
import sys
import os
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
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

# noinspection PyUnresolvedReferences
dtype = torch.FloatTensor

EPOCHS = 500
SEQ_LENGTH = 20
LEARNING_RATE = 0.001


DATA_TIME_STEPS = np.linspace(2, 10, SEQ_LENGTH + 1)
DATA = np.sin(DATA_TIME_STEPS)
DATA.resize((SEQ_LENGTH + 1, 1))

training_data = list(filter(None, read_from_stdin()))
# training_data = list(filter(None, read_training_data_from_file('tmp.txt')))
random.shuffle(training_data)
training_subset_size = 1100

ORDERED_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
alphabet = []
with open(r"data/tur_alphabet_wiki.txt", encoding='utf-8') as f:
    alphabet = [l.strip() for l in f]
vovels = [c for c in "aeiıoöüu"]
consonants = sorted(set(alphabet).difference(set(vovels)))


def onehot_encode_char(alphabet, char):
    onehot_encoding_char = ["0"] * len(alphabet)
    onehot_encoding_char[alphabet.index(char)] = "1"
    return onehot_encoding_char


x = Variable(torch.Tensor(DATA[:-1]).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(DATA[1:]).type(dtype), requires_grad=False)


input = torch.randn(3,3, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input)
print(target)
loss = nn.CrossEntropyLoss()

# print(DATA)

# noinspection PyUnresolvedReferences
def forward(input, context_units_state, w1, w2):
    xh = torch.cat((input, context_units_state), 1)
    context_units_state = torch.sigmoid(xh.mm(w1))
    out = context_units_state.mm(w2)
    return (out, context_units_state)


# noinspection PyUnresolvedReferences
def train(num_of_epochs):
    input_size = len(training_data[0][0][0]) + HIDDEN_NUM
    output_size = len(training_data[0][0][1])

    V = torch.FloatTensor(input_size, HIDDEN_NUM).type(dtype)
    init.normal_(V, 0.0, 0.4)
    V = Variable(V, requires_grad=True)

    # noinspection PyUnresolvedReferences
    W = torch.FloatTensor(HIDDEN_NUM, output_size).type(dtype)
    init.normal_(W, 0.0, 0.3)
    W = Variable(W, requires_grad=True)

    for i in range(num_of_epochs):
        random.shuffle(training_data)
        epoch_losses = []

        training_data_subset = training_data[:training_subset_size]
        val_data_subset = training_data[training_subset_size:]

        # print(training_data_subset)
        total_loss = []
        # noinspection PyUnresolvedReferences
        context_state = Variable(torch.zeros((1, HIDDEN_NUM)).type(dtype), requires_grad=True)

        for word in training_data_subset:

            batch_pred = []
            batch_target = []


            for letter in word:
                input = Variable(torch.Tensor([letter[0]]).type(dtype), requires_grad=False)
                target = Variable(torch.Tensor([letter[1].index(1)]).type(torch.LongTensor), requires_grad=False)

                (pred, context_state) = forward(input, context_state, V, W)
                batch_pred.append(pred)
                batch_target.append(target)

                context_state = Variable(context_state.data)

                output = loss(pred, target)  # (pred - target).pow(2).sum() / 2
                output.backward()
                total_loss.append(output.item())

            V.data -= LEARNING_RATE * V.grad.data
            W.data -= LEARNING_RATE * W.grad.data

            V.grad.data.zero_()
            W.grad.data.zero_()

        print("epoch {}, epoch avg loss: {}".format(i, np.mean(total_loss)))


    names = []
    values = [[] for _ in range(HIDDEN_NUM)]

    hidden_dict = [{} for _ in range(HIDDEN_NUM)]

    for letter in alphabet:

        input = Variable(torch.Tensor([[int(bit) for bit in onehot_encode_char(alphabet, letter)]]))

        (pred, context_state) = forward(input, context_state, V, W)

        for i in range(HIDDEN_NUM):
            hidden_dict[i][letter] = context_state.detach().numpy()[0][i]

    for name in ORDERED_ALPHABET:
        names.append(r'$\mathit{' + name + '}$')
        for i in range(HIDDEN_NUM):
            values[i].append(hidden_dict[i][name])

    for row in range(HIDDEN_NUM):
        plt.ylim(-0.2, 1.2)
        plt.plot(names, values[row], "o")

        for x, y in zip(names, values[row]):
            plt.annotate(x, xy=(x, y+0.05))

        plt.savefig(PLOTS_DIRNAME + '/unit_' + str(row) + '_e' + str(num_of_epochs) + 'torch.png')

        plt.clf()


train(EPOCHS)

# noinspection PyUnresolvedReferences


# def predict_eval(context_units):
#     for i in range(x.size(0)):
#         input = x[i:i + 1]
#         (pred, context_units) = forward(input, context_units, V, W)
#         context_units = context_units
#         predictions.append(pred.data.numpy().ravel()[0])


# predict_eval(context_units)

# pl.scatter(data_time_steps[:-1], x.data.numpy(), s=90, label="Actual")
# pl.scatter(data_time_steps[1:], predictions, label="Predicted")
# pl.legend()
# pl.show()
