from math import remainder
import os

from PIL.Image import init, new
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from operator import le
import numpy as np
from numpy.core.fromnumeric import size
from scipy.ndimage.interpolation import map_coordinates
import tensorflow as tf
import keras
from keras import layers
from memory_buffer import MemoryBuffer
import cppfunctions
import data_handling
from neural_network_util import set_layer_weights
from constants import *
from display_util import position_to_grid, add_tuple
from keras import Model






# agent brain, handles all info 
# processing and decision making.
class AgentBrain:
    
    def __init__(self):
        self.memory = MemoryBuffer(NUM_MEMORIES, ST_MEM_SIZE, INPUT_SHAPE)
        self.last_frames = [np.zeros(shape=INPUT_SHAPE) for _ in range(PREDICTION_FRAMES)]
        
        # init variables to be changed later during runtime
        self.weights_initted = False
        self.autoencoder = None
        self.autoencoder2 = None
        self.encoder = None
        self.decoder = None
        self.initial_filters = None
        
        # init storage and variables
        self.latent_classes = 12
        
        self.internal_map = \
            np.zeros(
                shape=(MAP_SIZE, MAP_SIZE, self.latent_classes), 
                dtype=bool)
        
        self.position = (0, 0)
        
        center = int(MAP_SIZE / 2)
        self.map_center = (center, center)
    
    
    # this function defines the autoencoder
    # that acts as an model for perceiving
    # the environment
    def construct_autoencoder(self, initial_filters):
        
        self.initial_filters = initial_filters
        
        autoencoder = keras.Sequential()
        encoder = keras.Sequential()
        decoder = keras.Sequential()
        
        # encoder layers
        autoencoder_input = layers.Input(shape=INPUT_SHAPE)
        encoder_input = layers.Input(shape=(8, 8, 1))
        autoencoder.add(autoencoder_input)
        encoder.add(encoder_input)
        
        
        layer = layers.Conv2D(
            initial_filters, (4, 4), strides=(4, 4),
            activation='relu', 
            name='layer1')
        autoencoder.add(layer)
        encoder.add(layer)
        
        
        # latent dimension
        layer = layers.Conv2D(
            self.latent_classes, (2, 2), strides=(2, 2),
            activation='sigmoid')
        autoencoder.add(layer)
        encoder.add(layer)
        
        
        # decoder layers
        decoder_input = keras.layers.Input(shape=self.internal_map.shape)
        decoder.add(decoder_input)
        
        layer = layers.Conv2DTranspose(
            initial_filters, (2, 2),  strides=(2, 2),
            activation='relu',
            name='conv_transpose1')
        autoencoder.add(layer)
        decoder.add(layer)
        
        
        layer = layers.Conv2DTranspose(
            1, (4, 4), strides=(4, 4),
            activation='sigmoid', 
            name='conv_out')
        autoencoder.add(layer)
        decoder.add(layer)
        
        self.autoencoder = autoencoder
        self.autoencoder2 = keras.models.clone_model(self.autoencoder)
        self.encoder = encoder
        self.decoder = decoder
        
        
        # compile it
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        loss = 'mean_squared_error'
        self.autoencoder.compile(loss=loss, optimizer=optimizer)
        self.autoencoder2.compile(loss=loss, optimizer=optimizer)
        
        self.encoder.compile()
        self.decoder.compile()
    
    
    def set_mapping_kernel(self):
        for layer in self.autoencoder.layers:
            if layer.name == 'layer1':
                self.mapping.set_weights(layer.get_weights())
    
    
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
                    self.autoencoder2.train_on_batch(x, x)
                end_i = num_batches * batch_size
                if end_i < len(data):
                    x = data[end_i:]
                    self.autoencoder.train_on_batch(x, x)
                    self.autoencoder2.train_on_batch(x, x)
        else:
            self.autoencoder.fit(data, data, epochs=epochs)
            self.autoencoder2.fit(data, data, epochs=epochs)
    
    
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
                x = data_handling.augment_data(x, 16, 16)
                
                # get the data
                data = cppfunctions.images_to_matrix(x, 4, 4)
                
                data = data_handling.select_data(data, 2000)
                
                # use it for training
                groups = data_handling.find_absolute_groups(data, 80, 120, 100, 2, 100)
                weights = cppfunctions.create_weights(groups)
                weights = np.reshape(weights, newshape=(weights.shape[0], 4, 4))
                
                self.construct_autoencoder(100)
                
                set_layer_weights(self.autoencoder, weights, 'layer1')
                
                x = np.reshape(x, newshape=(*x.shape, 1))
                
                self.train_internal_model(x, epochs=1)
                
                return groups
    
    
    def build_map(self, img):
        # how the real size translates to internal map size:
        # literal -> compressed input -> latent
        # 32x32, to NN input -> 4x4, through encoder -> 1x1
        # therfore, a 32x32 section on the screen is
        # described by a 1x1 cell of the grid.
        map_cell_size = 32
        
        map_x, map_y, adj_x, adj_y = position_to_grid(self.position, map_cell_size)
        map_x, map_y = add_tuple(self.map_center, (map_x, map_y))
        
        i = int(adj_x / 8)
        j = int(adj_y / 8)
        x = img[i:i+8, j:j+8]
        latent_vals = self.feed_encoder(x)
        print(latent_vals.shape)
        self.internal_map[map_x, map_y] = latent_vals
        return latent_vals is not None
    
    
    def input_autoencoder(self, inputs):
        hit_end = self.memory.insert_memory(inputs)
        if hit_end:
            x = data_handling.augment_data(self.memory.memory, 16, 16)
            x = np.reshape(x, newshape=(*x.shape, 1))
            self.train_internal_model(x)
    
    
    def process_inputs(self, inputs, img, change_in_position):
        self.position = add_tuple(self.position, change_in_position)
        if self.weights_initted is False:
            self.learn_groups(inputs)
        else:
            self.build_map(img)
            self.input_autoencoder(inputs)
    
    
    def reconstruct_internal_model(self, inputs):
        if self.autoencoder is not None:
            return self.autoencoder.predict(np.array([inputs]))[0]
    
    
    def reconstruct_internal_model2(self, inputs):
        if self.autoencoder2 is not None:
            return self.autoencoder2.predict(np.array([inputs]))[0]
    
    
    def feed_encoder(self, inputs):
        if self.encoder is not None:
            return self.encoder.predict(np.array([inputs]))[0]
    
    
    def read_map(self):
        inputs = np.array([self.internal_map], dtype=np.float32)
        outputs = self.decoder.predict(inputs)
        return outputs[0]
        
    
    
    def save_model(self):
        self.autoencoder.save('internal_model')
    
    
    def load_model(self):
        self.autoencoder = keras.models.load_model('internal_model')

if __name__ == "__main__":
    brain = AgentBrain()
    brain.construct_autoencoder(100)
