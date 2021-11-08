from operator import le
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
from set_weights import set_layer_weights


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

INITIAL_LAYER_NAME = 'layer1'



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
        #self.construct_predictive_model()
        self.memory = MemoryBuffer(NUM_MEMORIES, ST_MEM_SIZE, INPUT_SHAPE)
        self.eval_counter = 0
        self.last_frames = [np.zeros(shape=INPUT_SHAPE) for _ in range(PREDICTION_FRAMES)]
    
    
    def construct_internal_model(self):
        il = layers.Input(shape=INPUT_SHAPE)
        layer = layers.Conv2D(
            16, (4, 4), activation='relu', 
            name=INITIAL_LAYER_NAME)(il)
        layer = layers.Flatten()(layer)
        layer = layers.Dense(16, activation='relu')(layer)
        layer = layers.Dense(13 * 13 * 16)(layer)
        layer = layers.Reshape(target_shape=(13, 13, 16))(layer)
        layer = layers.Conv2DTranspose(3, (4, 4), activation='sigmoid')(layer)
        
        self.internal_model = keras.Model(inputs=il, outputs=layer)
        
        loss = keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE)
        
        self.internal_model.compile(loss=loss, optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
        self.internal_model.summary()
    
    
    def construct_predictive_model(self):
        latent_size = 16
        
        il = layers.Input(shape=(PREDICTION_FRAMES, *INPUT_SHAPE))
        layer = layers.Conv3D(8, (2, 4, 4), activation='relu')(il)
        layer = layers.Flatten()(layer)
        layer = layers.Dense(latent_size, activation='relu')(layer)
        layer = layers.Dense(13 * 13 * 8)(layer)
        layer = layers.Reshape(target_shape=(13, 13, 8))(layer)
        layer = layers.Conv2DTranspose(3, (4, 4), activation='sigmoid')(layer)
        self.pred_model = keras.Model(inputs=il, outputs=layer)
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.pred_model.compile(optimizer=optimizer, loss="binary_crossentropy")
    
    
    def construct_location_model(self):
        il = layers.Input(shape=INPUT_SHAPE)
        layer = layers.Conv2D(16, (4, 4), activation='relu')(il)
        layer = layers.Flatten()(layer)
        layer = layers.Dense(16, activation='relu')(layer)
        
        self.location_model = keras.Model(inputs=il, outputs=layer)
        
        loss = keras.losses.MeanSquaredError(reduction=tf.compat.v1.losses.Reduction.NONE)
        
        self.location_model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=0.0001))
        self.location_model.summary()
    
    
    def forward_process_(self, inputs):
        self.last_frames = self.last_frames[1:]
        self.last_frames.append(inputs)
        print("added memory")
        hit_end = self.memory.insert_memory(inputs)
        if hit_end:
            x, y = get_sequences(self.memory.memory, PREDICTION_FRAMES)
            print("training on memories...")
            self.pred_model.fit(x, y, shuffle=True)
    
    
    def forward_process(self, inputs):
        same_as_last = self.memory.equals_last(inputs)
        
        # only add new data to memories
        if not same_as_last:
            hit_end = self.memory.insert_memory(inputs)
            
            # when you reach capacity
            if hit_end:
                # get the data
                data = cppfunctions.images_to_matrix_4d(self.memory.memory, 4, 4)
                
                new_data = []
                for i in range(len(data)):
                    if np.linalg.norm(data[i]) >= 0.1:
                        new_data.append(data[i])
                data = np.array(new_data, dtype=np.float32)
                
                # augment the data
                data = cppfunctions.augment_data(data, 4, 4)
                data = np.reshape(
                    data, newshape=(data.shape[0], int(data.size / data.shape[0])))
                
                indices = np.argsort(np.linalg.norm(data, axis=1))
                data = data[indices]
                data = data[-2000:]
                
                # use it for training
                groups = grouping_data.find_absolute_groups(data, 300, 120, 100, 2, 100)
                weights = cppfunctions.create_weights(groups)
                
                #set_layer_weights(self.internal_model, weights, INITIAL_LAYER_NAME)
                
                return groups
    
    
    def reconstruct_internal_model(self, inputs):
        return self.internal_model.predict(np.array([inputs]))[0]
    
    
    def pred_next_frame(self):
        return self.pred_model.predict(np.array([self.last_frames]))[0]
    
    
    def save_model(self):
        self.internal_model.save('internal_model')
    
    
    def load_model(self):
        self.internal_model = keras.models.load_model('internal_model')


