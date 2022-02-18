from enum import auto
import os

from neural_network_util import set_layer_weights
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from operator import le
import numpy as np
import keras
from keras import layers
from memory_buffer import MemoryBuffer
import cppfunctions
import data_handling
from constants import *
from display_util import add_tuple, to_greyscale
from keras import Model, Sequential
import math
from calculate_map_edges import *
import cv2





# agent brain, handles all info 
# processing and decision making.
class AgentBrain:
    
    def __init__(self):
        self.memory = MemoryBuffer(NUM_MEMORIES, ST_MEM_SIZE, INPUT_SHAPE)
        self.last_frames = [np.zeros(shape=INPUT_SHAPE) for _ in range(PREDICTION_FRAMES)]
        
        # init variables to be changed later during runtime
        self.weights_initted = False
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.small_autoencoder = None
        self.initial_filters = None
        
        # init storage and variables
        self.latent_classes = None
        self.internal_map = None
        self.position = (0, 0)
        
        center = int(MAP_SIZE / 2)
        self.map_center = (center, center)
        self.map_input = None
        self.map_outputs = None
        self.map_data = []
    
    
    def init_internal_map(self):
        self.internal_map = \
            np.zeros(
                shape=(MAP_SIZE, MAP_SIZE, self.latent_classes), 
                dtype=bool)
    
    
    # this function defines the autoencoder
    # that acts as an model for perceiving
    # the environment
    def construct_autoencoder(self, initial_filters):
        
        self.initial_filters = initial_filters
        
        autoencoder = Sequential()
        encoder = Sequential()
        decoder = Sequential()
        
        # encoder layers
        autoencoder_input = layers.Input(shape=INPUT_SHAPE)
        encoder_input = layers.Input(shape=(8, 8, 1))
        autoencoder.add(autoencoder_input)
        encoder.add(encoder_input)
        
        
        layer = layers.Conv2D(
            initial_filters, (4, 4), strides=(4, 4),
            activation='relu', 
            name='conv_in')
        autoencoder.add(layer)
        encoder.add(layer)
        
        '''
        # latent dimension
        layer = layers.Conv2D(
            self.latent_classes, (2, 2), strides=(2, 2),
            activation='sigmoid')
        autoencoder.add(layer)
        encoder.add(layer)
        '''
        
        # decoder layers
        decoder_input = keras.layers.Input(shape=(1, 1, self.latent_classes))
        decoder.add(decoder_input)
        
        '''
        layer = layers.Conv2DTranspose(
            initial_filters, (2, 2),  strides=(2, 2),
            activation='relu',
            name='conv_transpose1')
        autoencoder.add(layer)
        decoder.add(layer)
        '''
        
        layer = layers.Conv2DTranspose(
            1, (4, 4), strides=(4, 4),
            activation='sigmoid', 
            name='conv_out')
        autoencoder.add(layer)
        decoder.add(layer)
        
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        
        # compile it
        optimizer = keras.optimizers.RMSprop(learning_rate=0.005)
        loss = 'mean_squared_error'
        self.autoencoder.compile(loss=loss, optimizer=optimizer)
        
        self.encoder.compile()
        self.decoder.compile()
        
        self.small_autoencoder = Sequential()
        self.small_autoencoder.add(self.encoder)
        self.small_autoencoder.add(self.decoder)
        self.small_autoencoder.compile()
    
    
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
                
                data = data_handling.select_data(data, 100)
                
                # use it for training
                groups = data_handling.find_groups(data, 4, 100, 9.0, 1.0, stop_at=64)
                print("clustering finished.")
                input_weights = cppfunctions.create_weights(groups)
                input_weights = np.reshape(input_weights, newshape=(input_weights.shape[0], 4, 4))
                output_weights = np.array(groups, dtype=np.float32)
                output_weights = np.reshape(output_weights, newshape=(output_weights.shape[0], 4, 4))
                
                # once the number of groups is known, do this
                self.latent_classes = len(groups)
                print("latent classes: ", self.latent_classes)
                self.init_internal_map()
                self.construct_autoencoder(self.latent_classes)
                
                set_layer_weights(self.autoencoder, input_weights, 'conv_in')
                set_layer_weights(self.autoencoder, output_weights, 'conv_out')
                
                x = np.reshape(x, newshape=(*x.shape, 1))
                
                #self.train_autoencoder(x, epochs=1, use_model_fit=True)
                
                return groups
    
    
    def train_autoencoder(
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
            self.autoencoder.evaluate(x, x)
        else:
            self.autoencoder.fit(
                data, data, epochs=epochs, batch_size=batch_size)
    
    
    def get_map_image(self, pos, view):
        x, y = pos
        vx = view.shape[0]
        vy = view.shape[1]
        
        # position of
        # grid cell in view = edge of grid cell - edge of view
        grid_x = (CELL_SIZE * math.floor(x / CELL_SIZE))
        view_x = x - (vx / 2)
        x = int(grid_x - view_x)
        
        grid_y = (CELL_SIZE * math.floor(y / CELL_SIZE))
        view_y = y - (vy / 2)
        y = int(grid_y - view_y)
        
        return view[x:x+CELL_SIZE, y:y+CELL_SIZE]
    
    
    def get_grid_location(self):
        map_x = int(self.position[0] / CELL_SIZE)
        map_y = int(self.position[1] / CELL_SIZE)
        return map_x, map_y
    
    
    # encode something to the map
    def encode_to_map(self, view_img):
        xmin, xmax, ymin, ymax = map_edges(self.position, view_img.shape, 64)
        
        img = view_img[xmin:xmax, ymax:ymin, :]
        
        resized = cv2.resize(to_greyscale(img), (8, 8))
        img = np.reshape(resized, newshape=(8, 8, 1))
        
        self.map_input = img
        img = np.array([img], dtype=np.float32)
        
        latent_vector = self.encoder.predict(img)
        
        if latent_vector is not None and False:
            self.internal_map[map_x, map_y] = latent_vector[0, 0, 0]
            #x = np.array(latent_vector)
            #self.map_outputs = self.decoder.predict(x)[0]
        return latent_vector is not None
    
    
    def input_autoencoder(self, inputs):
        hit_end = self.memory.insert_memory(inputs)
        if hit_end:
            x = data_handling.augment_data(self.memory.memory, 16, 16)
            x = np.reshape(x, newshape=(*x.shape, 1))
            self.train_autoencoder(x, epochs=2)
    
    
    # take inputs passed to the agent
    # this is the equivalent of perception
    def process_inputs(self, inputs, img, change_in_position):
        self.position = add_tuple(self.position, change_in_position)
        if self.weights_initted is False:
            self.learn_groups(inputs)
        else:
            self.encode_to_map(img)
            self.input_autoencoder(inputs)
    
    
    def reconstruct_internal_model(self, inputs):
        if self.autoencoder is not None:
            return self.autoencoder.predict(np.array([inputs]))[0]
    
    
    def feed_encoder(self, inputs):
        if self.encoder is not None:
            return self.encoder.predict(np.array([inputs]))[0]
    
    
    def read_map(self):
        return self.map_outputs
    
    
    def save_model(self):
        self.autoencoder.save('internal_model')
    
    
    def load_model(self):
        self.autoencoder = keras.models.load_model('internal_model')


if __name__ == "__main__":
    brain = AgentBrain()
    brain.construct_autoencoder(100)
