import torch
from torch.autograd import Variable
import numpy as np

import torch.nn as nn
import torch.nn.init as init


class ModelPyTorch:
    dtype = torch.FloatTensor
    crossEntropyLoss = nn.CrossEntropyLoss()

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        V = torch.FloatTensor(self._input_dim + self._hidden_dim, self._hidden_dim).type(self.dtype)
        init.normal_(V)
        self._V = Variable(V, requires_grad=True)

        W = torch.FloatTensor(self._hidden_dim, self._output_dim).type(self.dtype)
        init.normal_(W)
        self._W = Variable(W, requires_grad=True)

        # todo: init with normal
        self._context_state_vector = Variable(torch.zeros((1, hidden_dim)).type(self.dtype), requires_grad=True)

        self._learning_rate = learning_rate
        self._trainer = torch.optim.SGD((self._V, self._W), lr=self._learning_rate, momentum=0.9)

    def batch_forward(self, training_vectors):
        batch_loss = []
        predictions = []
        for input_vector, target_vector in training_vectors:
            pred = self._forward(input_vector)
            target = Variable(torch.Tensor([target_vector.index(1)]).type(torch.LongTensor), requires_grad=False)

            loss = self.crossEntropyLoss(pred, target)

            batch_loss.append(loss)
            predictions.append(pred.detach().numpy()[0])

        return predictions, batch_loss, np.mean([loss.item() for loss in batch_loss])

    def _forward(self, input_vector):
        input_vector = Variable(torch.Tensor([input_vector]).type(self.dtype), requires_grad=False)
        xh = torch.cat((input_vector, self._context_state_vector), 1)
        self._context_state_vector = torch.sigmoid(xh.mm(self._V))
        output_values = self._context_state_vector.mm(self._W)

        self._context_state_vector = Variable(self._context_state_vector.data)

        return output_values

    def train_batches(self, training_batches):
        return self._process_batches(training_batches, learning=True)

    def predict_batches(self, training_batches):
        return self._process_batches(training_batches, learning=False)

    def _process_batches(self, training_batches, learning: bool):
        epoch_loss = []
        predictions = []
        for training_batch in training_batches:
            batch_predictions, losses, mean_loss = self.batch_forward(training_batch)

            if learning:
                self._process_loss(losses)
                self._trainer.step()
                self._trainer.zero_grad()

            epoch_loss.append(mean_loss)
            predictions.append(batch_predictions)

        return predictions, np.mean(epoch_loss)

    @staticmethod
    def _process_loss(losses):
        for loss in losses:
            loss.backward()

    def get_context_state(self):
        return self._context_state_vector.numpy()[0]
