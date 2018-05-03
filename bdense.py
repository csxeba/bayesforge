from keras.layers import Dense
from keras import backend as K

from bayesify import BayesifierMixin


class BayesianDense(BayesifierMixin, Dense):

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 variation_initializer="ones",
                 variation_regularizer=None,
                 variation_constraint=None,
                 **kwargs):
        Dense.__init__(self,
                       units=units,
                       activation=activation,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       activity_regularizer=activity_regularizer,
                       kernel_constraint=kernel_constraint,
                       bias_constraint=bias_constraint,
                       **kwargs)
        BayesifierMixin.__init__(self,
                                 variational_initializer=variation_initializer,
                                 variational_regularizer=variation_regularizer,
                                 variational_constraint=variation_constraint)

    def call(self, inputs, **kwargs):
        param = K.in_train_phase(self.sample_weights(), self.mean)
        output = K.dot(inputs, param[0])
        if self.use_bias:
            output = K.bias_add(output, param[1])
        if self.activation:
            output = self.activation(output)
        return output
