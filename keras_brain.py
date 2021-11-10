from operator import le
from os import name
from random import shuffle
from PIL.Image import new
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.python.keras.backend import dtype
from memory_buffer import MemoryBuffer
import cppfunctions
import grouping_data
from neural_network_util import set_layer_weights


INPUT_SHAPE = (16, 16, 1)
INPUT_SIZE = 1
for dim in INPUT_SHAPE:
    INPUT_SIZE *= dim
NUM_ACTIONS = 3

# the number of "memories" to use for training
NUM_MEMORIES = 1000

# short term memory size.
# this should be at least the number
# of the highest derivative of motion
# the network should be able to detect
ST_MEM_SIZE = 4
NUM_EARLY_MEMORIES = 256
PREDICTION_FRAMES = 4


def create_rand_arr(shape):
    initializer = keras.initializers.GlorotUniform()
    return initializer(shape=shape)


def get_sequences(data, seq_len):
    size = len(data) - seq_len
    dshape = data[0].shape
    seqs = np.zeros(shape=(size, seq_len, *dshape))
    results = np.zeros(shape=(size, *dshape))
    for i in range(size):
        seqs[i] = data[i:i+seq_len]
        results[i] = data[i+seq_len]
    return seqs, results


# agent brain, handles all info 
# processing and decision making.
class AgentBrain:
    
    def __init__(self):
        self.memory = MemoryBuffer(NUM_MEMORIES, ST_MEM_SIZE, INPUT_SHAPE, 16)
        self.eval_counter = 0
        self.last_frames = [np.zeros(shape=INPUT_SHAPE) for _ in range(PREDICTION_FRAMES)]
        self.weights_initted = False
        self.internal_model = None
    
    
    def construct_internal_model(self, initial_filters):
        il = layers.Input(shape=INPUT_SHAPE)
        layer = layers.Conv2D(
            initial_filters, (4, 4), 
            activation='relu', 
            name='layer1')(il)
        layer = layers.Conv2D(
            16, (4, 4), 
            activation='relu')(layer)
        layer = layers.Dense(1, activation='relu')(layer)
        layer = layers.Dense(16, activation='relu')(layer)
        
        layer = layers.Conv2DTranspose(
            initial_filters, (4, 4), 
            activation='relu')(layer)
        
        layer = layers.Conv2DTranspose(
            1, (4, 4), 
            activation='sigmoid', 
            name='conv_out')(layer)
        
        self.internal_model = keras.Model(inputs=il, outputs=layer)
        
        loss = keras.losses.MeanSquaredError(
            reduction=tf.compat.v1.losses.Reduction.NONE)
        
        self.internal_model.compile(
            loss=loss, optimizer=keras.optimizers.RMSprop(
                learning_rate=0.001))
        
        self.internal_model.summary()
    
    
    def construct_grid_model(self):
        im = self.internal_model
        im.trainable = False
        
        model = keras.Sequential()
        model.add(im)
        model.add(layers.Conv2D(16, (4, 4), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        
        self.grid_model = model
        
        loss = keras.losses.MeanSquaredError(
            reduction=tf.compat.v1.losses.Reduction.NONE)
        
        self.grid_model.compile(
            loss=loss, 
            optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.grid_model.summary()
    
    
    def train_internal_model(self, data, epochs=10):
        self.internal_model.fit(data, data, epochs=epochs)
    
    
    def train_grid_model(self, epochs=10):
        self.grid_model.fit(
            self.memory.memory, 
            self.memory.grid_memory, 
            epochs=epochs)
    
    
    def add_last_frame(self, inputs):
        self.last_frames = self.last_frames[1:]
        self.last_frames.append(inputs)
    
    
    def forward_process_learning(self, inputs):
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
                
                print(x.shape)
                
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
                #self.construct_grid_model()
                
                set_layer_weights(self.internal_model, weights, 'layer1')
                set_layer_weights(self.internal_model, weights, 'conv_out')
                
                x = np.reshape(x, newshape=(*x.shape, 1))
                
                self.train_internal_model(x)
                
                return groups
    
    
    def forward_process(self, inputs, grid_activations):
        if self.weights_initted is False:
            self.forward_process_learning(inputs)
            return None
        else:
            hit_end = self.memory.insert_memory(inputs, grid_activations)
            if hit_end:
                x = cppfunctions.augment_data(self.memory.memory, 16, 16)
                x = np.reshape(x, newshape=(*x.shape, 1))
                self.train_internal_model(x)
                self.train_grid_model()
    
    
    def reconstruct_internal_model(self, inputs):
        if self.internal_model is not None:
            return self.internal_model.predict(np.array([inputs]))[0]
    
    
    def pred_next_frame(self):
        return self.pred_model.predict(np.array([self.last_frames]))[0]
    
    
    def save_model(self):
        self.internal_model.save('internal_model')
    
    
    def load_model(self):
        self.internal_model = keras.models.load_model('internal_model')
