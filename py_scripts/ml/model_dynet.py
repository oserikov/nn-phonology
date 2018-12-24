import dynet as dy
import numpy as np


class ModelDyNet:
    _input_l = dy.Expression
    _context_l = dy.Expression
    _output_l = dy.Expression

    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        self._model = dy.ParameterCollection()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim

        self._V = self._model.add_parameters((self._hidden_dim, self._input_dim + self._hidden_dim), init=dy.NormalInitializer())
        # noinspection PyRedundantParentheses
        self._b = self._model.add_parameters((self._hidden_dim), init=dy.ConstInitializer(0))
        self._W = self._model.add_parameters((self._output_dim, self._hidden_dim), init=dy.NormalInitializer())

        self._context_state_vector = dy.random_normal(hidden_dim).npvalue()

        self._learning_rate = learning_rate
        self._trainer = dy.MomentumSGDTrainer(self._model, learning_rate=self._learning_rate)

        self._init_layers()

    def _batch_forward(self, training_vectors):
        batch_loss = []
        predictions = []
        for input_vector, target_vector in training_vectors:
            pred = dy.softmax(self._forward(input_vector))
            loss = -dy.tanh(dy.pick(pred, target_vector.index(1)))
            batch_loss.append(loss)
            predictions.append(pred.npvalue())

        return predictions, dy.esum(batch_loss), np.mean([loss.npvalue() for loss in batch_loss])

    def _forward(self, input_vector):
        input_vector.extend(self._context_state_vector)
        self._input_l.set(input_vector)

        output_values = self._output_l

        self._context_state_vector = self._context_l.npvalue()

        return output_values

    def train_batches(self, training_batches):
        return self._process_batches(training_batches, learning=True)

    def predict_batches(self, training_batches):
        return self._process_batches(training_batches, learning=False)

    def _process_batches(self, training_batches, learning: bool):
        epoch_loss = []
        predictions = []
        for training_batch in training_batches:
            self._reload_model_into_cg()

            batch_predictions, loss, mean_loss = self._batch_forward(training_batch)

            if learning:
                self._process_loss(loss)
                self._trainer.update()

            epoch_loss.append(mean_loss)
            predictions.append(batch_predictions)

        return predictions, np.mean(epoch_loss)

    @staticmethod
    def _process_loss(loss):
        loss.backward()

    def _reload_model_into_cg(self):
        dy.renew_cg()
        self._init_layers()
        # pass

    def _init_layers(self):
        self._input_l = dy.vecInput(self._input_dim + self._hidden_dim)
        self._context_l = dy.logistic(self._V * self._input_l + self._b)
        self._output_l = self._W * self._context_l

    def get_context_state(self):
        return self._context_state_vector
