from keras.layers.wrappers import Wrapper
from keras import backend as K
from keras.regularizers import Regularizer

K.set_learning_phase(1)


class VariationalRegularizer(Regularizer):

    def __init__(self, coef):
        self.coef = coef * 0.5

    def __call__(self, variation):
        return self.coef * K.sum(variation + K.log(variation) - 1.)


class Bayesify(Wrapper):

    def __init__(self, base_layer,
                 variational_initializer="ones",
                 variational_regularizer=None,
                 variational_constraint=None,
                 **kwargs):
        super().__init__(base_layer, **kwargs)
        self.variational_initializer = variational_initializer
        self.variational_regularizer = variational_regularizer or VariationalRegularizer(coef=0.1)
        self.variational_constraint = variational_constraint
        self.mean = []
        self.variation = []

    def build(self, input_shape=None):
        super().build(input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.mean = self.layer.trainable_weights[:]
        for tensor in self.layer.get_weights():
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
        for tensor, param in zip(self.layer.trainable_weights, sample):
            K.update(tensor, param)
        return self.layer.call(inputs, **kwargs)

    def __getattr__(self, item):
        return self.layer.__getattr__(item)
