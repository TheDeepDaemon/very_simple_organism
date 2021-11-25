import os
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
        self.eval_counter = 0
        self.last_frames = [np.zeros(shape=INPUT_SHAPE) for _ in range(PREDICTION_FRAMES)]
        self.weights_initted = False
        self.autoencoder = None
        self.autoencoder2 = None
        self.map_encoder = None
        self.latent_classes = 12
        self.internal_map = np.zeros(shape=(MAP_SIZE, MAP_SIZE, self.latent_classes), dtype=bool)
        self.map_tracker = np.zeros(shape=(MAP_SIZE, MAP_SIZE))
        self.position = (0, 0)
        self.map_size = MAP_SIZE
        center = int(MAP_SIZE / 2)
        self.map_center = (center, center)
    
    
    # this function defines the autoencoder
    # that acts as an model for perceiving
    # the environment
    def construct_autoencoder(self, initial_filters):
        
        # encoder layers
        il = layers.Input(shape=INPUT_SHAPE)
        layer = layers.Conv2D(
            initial_filters, (4, 4), strides=(4, 4),
            activation='relu', 
            name='layer1')(il)
        
        # compress into latent dimension
        layer = layers.Conv2D(
            self.latent_classes, (2, 2), strides=(2, 2),
            activation='sigmoid')(layer)
        
        # just the encoder
        self.encoder = Model(inputs=il, outputs=layer)
        
        # decoder layers
        layer = layers.Conv2DTranspose(
            initial_filters, (2, 2),  strides=(2, 2),
            activation='relu')(layer)
        
        layer = layers.Conv2DTranspose(
            1, (4, 4), strides=(4, 4),
            activation='sigmoid', 
            name='conv_out')(layer)
        
        
        # create the model
        self.autoencoder = Model(inputs=il, outputs=layer)
        
        self.autoencoder2 = keras.models.clone_model(self.autoencoder)
        
        optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        
        #loss = 'binary_crossentropy'
        loss = 'mean_squared_error'
        
        self.autoencoder.compile(loss=loss, optimizer=optimizer)
        
        self.autoencoder2.compile(loss=loss, optimizer=optimizer)
        
        
        il = keras.layers.Input(shape=(8, 8, 1))
        
        layer = layers.Conv2D(
            initial_filters, (4, 4), strides=(4, 4),
            activation='relu', 
            name='layer1')(il)
        
        layer = layers.Conv2D(
            self.latent_classes, (2, 2), strides=(2, 2),
            activation='sigmoid')(layer)
        
        self.map_encoder = Model(inputs=il, outputs=layer)
        self.map_encoder.compile()
    
    
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
                set_layer_weights(self.autoencoder, weights, 'conv_out')
                
                x = np.reshape(x, newshape=(*x.shape, 1))
                
                self.train_internal_model(x, epochs=1)
                
                return groups
    
    
    def build_map(self, img, map_row, map_col):
        x = img[:8,:8]
        latent_vals = self.feed_map(x)
        self.internal_map[map_row, map_col] = latent_vals
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
            
            # how the real size translates to internal map size:
            # literal -> compressed input -> latent
            # 32x32, to NN input -> 4x4, through encoder -> 1x1
            # therfore, a 32x32 section on the screen is
            # described by a 1x1 cell of the grid.
            map_cell_size = 32
            
            map_col, map_row = position_to_grid(self.position, map_cell_size)
            map_col, map_row = add_tuple(self.map_center, (map_col, map_row))
            
            self.build_map(img, map_row, map_col)
            self.input_autoencoder(inputs)
    
    
    def reconstruct_internal_model(self, inputs):
        if self.autoencoder is not None:
            return self.autoencoder.predict(np.array([inputs]))[0]
    
    
    def reconstruct_internal_model2(self, inputs):
        if self.autoencoder2 is not None:
            return self.autoencoder2.predict(np.array([inputs]))[0]
    
    
    def feed_map(self, inputs):
        if self.map_encoder is not None:
            return self.map_encoder.predict(np.array([inputs]))[0]
    
    
    def save_model(self):
        self.autoencoder.save('internal_model')
    
    
    def load_model(self):
        self.autoencoder = keras.models.load_model('internal_model')

if __name__ == "__main__":
    brain = AgentBrain()
    brain.construct_autoencoder(100)
