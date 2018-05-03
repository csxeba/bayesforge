from keras.layers.wrappers import Wrapper
from keras import backend as K

K.set_learning_phase(1)


class Bayesify(Wrapper):

    def __init__(self, base_layer,
                 variational_initializer="ones",
                 variational_regularizer=None,
                 variational_constraint=None,
                 **kwargs):
        super().__init__(base_layer, **kwargs)
        self.variational_initializer = variational_initializer
        self.variational_regularizer = variational_regularizer
        self.variational_constraint = variational_constraint
        self.mean = []
        self.variation = []

    def build(self, input_shape=None):
        super().build(input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.input_spec = self.layer.input_spec
        self.mean = self.layer.trainable_weights[:]
        for tensor in self.layer.trainable_weights:
            self.variation.append(self.add_weight(
                name="variation",
                shape=tensor.shape,
                initializer=self.variational_initializer,
                regularizer=self.variational_regularizer,
                constraint=self.variational_constraint
            ))

    def _sample_weights(self):
        return [mean + K.log(1. + K.exp(log_stddev))*K.random_normal(shape=mean.shape)
                for mean, log_stddev in zip(self.mean, self.variation)]

    def call(self, inputs, **kwargs):
        sample = K.in_train_phase(self._sample_weights(), self.mean)
        for tensor, parameter in zip(self.layer._trainable_weights, sample):
            K.set_value(tensor, parameter)
        return self.layer.call(inputs, **kwargs)


class BayesifierMixin:

    def __init__(self,
                 variational_initializer="ones",
                 variational_regularizer=None,
                 variational_constraint=None):
        self.variational_initializer = variational_initializer
        self.variational_regularizer = variational_regularizer
        self.variational_constraint = variational_constraint
        self.mean = []
        self.variation = []

    def build(self, input_shape=None):
        self.mean = self.trainable_weights[:]
        for tensor in self.mean:
            self.variation.append(self.add_weight(
                name="variation",
                shape=tensor.shape,
                initializer=self.variational_initializer,
                regularizer=self.variational_regularizer,
                constraint=self.variational_constraint
            ))

    def sample_weights(self):
        return [mean + K.log(1. + K.exp(log_stddev))*K.random_normal(shape=mean.shape)
                for mean, log_stddev in zip(self.mean, self.variation)]

    def call(self, inputs, **kwargs):
        raise NotImplementedError
