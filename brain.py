from operator import le
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from memory_buffer import MemoryBuffer
import cppfunctions
import grouping_data
from neural_network_util import set_layer_weights
from constants import *


rotation = 0

class Mapping(keras.layers.Layer):
    
    def __init__(self, units, input_dim, map_ref):
        super(Mapping, self).__init__()
        self.map = map_ref
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        # transform to map shape
        
        
        # embed in the map
        
        
        # get the outputs
        
        
        return tf.matmul(inputs, self.w) + self.b


# agent brain, handles all info 
# processing and decision making.
class AgentBrain:
    
    def __init__(self):
        self.memory = MemoryBuffer(NUM_MEMORIES, ST_MEM_SIZE, INPUT_SHAPE)
        self.eval_counter = 0
        self.last_frames = [np.zeros(shape=INPUT_SHAPE) for _ in range(PREDICTION_FRAMES)]
        self.weights_initted = False
        self.autoencoder = None
        self.internal_map = np.zeros(shape=(), dtype=bool)
        self.map_arr = np.zeros(shape=(MAP_SIZE, MAP_SIZE), dtype=np.int32)
    
    
    # this function defines the autoencoder
    # that acts as an model for perceiving
    # the environment
    def construct_internal_model(self, initial_filters):
        
        latent_dim_filters = 4
        
        # encoder layers
        il = layers.Input(shape=INPUT_SHAPE)
        layer = layers.Conv2D(
            initial_filters, (4, 4), 
            activation='relu', 
            name='layer1')(il)
        layer = layers.Conv2D(
            16, (4, 4), 
            activation='relu')(layer)
        
        # latent dimension
        layer = layers.Conv2D(
            latent_dim_filters, (2, 2), strides=(2, 2),
            activation='sigmoid')(layer)
        
        # store the encoder
        self.encoder = keras.Model(inputs=il, outputs=layer)
        
        # decoder layers
        # latent dimension
        layer = layers.Conv2DTranspose(
            latent_dim_filters, (2, 2), strides=(2, 2),
            activation='relu')(layer)
        
        layer = layers.Conv2DTranspose(
            initial_filters, (4, 4), 
            activation='relu')(layer)
        
        layer = layers.Conv2DTranspose(
            1, (4, 4), 
            activation='sigmoid', 
            name='conv_out')(layer)
        
        # create the model
        self.autoencoder = keras.Model(inputs=il, outputs=layer)
        
        loss = keras.losses.MeanSquaredError(
            reduction=tf.compat.v1.losses.Reduction.NONE)
        
        self.autoencoder.compile(
            loss=loss, optimizer=keras.optimizers.RMSprop(
                learning_rate=0.005))
        
        self.autoencoder.summary()
    
    
    def train_internal_model(
        self, data, epochs=1, 
        batch_size=16, use_model_fit=False):
        
        if not use_model_fit:
            for _ in range(epochs):
                np.random.shuffle(data)
                num_batches = int(len(data) / batch_size)
                for i_ in range(num_batches):
                    i = i_ * batch_size
                    x = data[i:i+batch_size]
                    self.autoencoder.train_on_batch(x, x)
                end_i = num_batches * batch_size
                if end_i < len(data):
                    x = data[end_i:]
                    self.autoencoder.train_on_batch(x, x)
        else:
            self.autoencoder.fit(data, data, epochs=epochs)
    
    
    def add_last_frame(self, inputs):
        self.last_frames = self.last_frames[1:]
        self.last_frames.append(inputs)
    
    
    def learn_groups(self, inputs):
        same_as_last = self.memory.equals_last(inputs)
        
        # only add new data to memories
        if not same_as_last:
            hit_end = self.memory.insert_memory(inputs)
            
            # when you reach capacity
            if hit_end:
                self.weights_initted = True
                x = self.memory.memory
                
                # augment the data by rotating and flipping
                x = cppfunctions.augment_data(x, 16, 16)
                
                # get the data
                data = cppfunctions.images_to_matrix(x, 4, 4)
                
                new_data = []
                for i in range(len(data)):
                    if np.linalg.norm(data[i]) >= 0.1:
                        new_data.append(data[i])
                data = np.array(new_data, dtype=np.float32)
                
                # sort based on norm to remove blank data, 
                # images that are all or mostly black will have low norms.
                indices = np.argsort(np.linalg.norm(data, axis=1))
                data = data[indices]
                data = data[-2000:]
                
                # use it for training
                groups = grouping_data.find_absolute_groups(data, 300, 120, 100, 2, 100)
                weights = cppfunctions.create_weights(groups)
                weights = np.reshape(weights, newshape=(weights.shape[0], 4, 4))
                
                self.construct_internal_model(weights.shape[0] * 2)
                
                set_layer_weights(self.autoencoder, weights, 'layer1')
                set_layer_weights(self.autoencoder, weights, 'conv_out')
                
                x = np.reshape(x, newshape=(*x.shape, 1))
                
                self.train_internal_model(x, epochs=3)
                
                return groups
    
    
    def process_inputs(self, inputs, pos_delta):
        
        if self.weights_initted is False:
            self.learn_groups(inputs)
            return None
        else:
            hit_end = self.memory.insert_memory(inputs)
            if hit_end:
                x = cppfunctions.augment_data(self.memory.memory, 16, 16)
                x = np.reshape(x, newshape=(*x.shape, 1))
                self.train_internal_model(x)
    
    
    def reconstruct_internal_model(self, inputs):
        if self.autoencoder is not None:
            return self.autoencoder.predict(np.array([inputs]))[0]
    
    
    def predict_grid_position(self, inputs):
        if self.grid_model is not None:
            return self.grid_model.predict(np.array([inputs]))[0]
    
    
    def save_model(self):
        self.autoencoder.save('internal_model')
    
    
    def load_model(self):
        self.autoencoder = keras.models.load_model('internal_model')
