import numpy as np
from numpy.lib.function_base import percentile
import tensorflow as tf
import cppfunctions


def create_matrix(rows, cols):
    initializer = tf.keras.initializers.GlorotUniform()
    return initializer(shape=(rows, cols))


class Perceptron:
    
    def __init__(self, size):
        self.weights = create_matrix(rows=size, cols=10)
    
    def mul(self, inputs):
        return tf.linalg.matvec(self.weights, inputs)
    
    def forward_process(self, inputs):
        return tf.keras.activations.sigmoid(self.mul(inputs))
    
    def learn_patterns(self, data):
        pass
    
    def backwards_process(self, inputs, outputs):
        pass

