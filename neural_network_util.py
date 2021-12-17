import numpy as np
import keras


def add_rows_to_dense(model, weights_index, rows):
    '''modify weights of a keras model to add these rows'''
    mats = model.get_weights()
    mat = mats[weights_index]
    mat = np.append(mat, np.array(rows), axis=0)
    mat = np.transpose(mat, axes=(1, 0))
    mats[weights_index] = mat
    bias_index = weights_index+1
    mats[bias_index] = np.append(mats[bias_index], np.zeros(len(rows)))
    mats[bias_index+1].resize((mats[bias_index+1].shape[0]+len(rows), mats[bias_index+1].shape[1]))
    mats[weights_index+2][-1].fill(0.)
    return mats



# replace the kernel of a conv2d layer
def replace_kernel(weights, kernel, filter_index, channel_index=None):
    shape = kernel.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            
            # if there are numtiple input channels
            if channel_index is None:
                for c in range(len(weights[0][i][j])):
                    weights[0][i, j, c, filter_index] = kernel[i, j]
            else:
                weights[0][i, j, channel_index, filter_index] = kernel[i, j]


def set_layer_weights(model, kernels, layer_name):
    for layer in model.layers:
        weights = layer.get_weights()
        if layer.name == layer_name:
            for i, kernel in enumerate(kernels):
                replace_kernel(weights, kernel, i)
            layer.set_weights(weights)
            break


# use GlorotUniform to make a random array of values
def create_rand_arr(shape):
    initializer = keras.initializers.GlorotUniform()
    return initializer(shape=shape)


# get sequences of a few frames in a row
# that can be used to train the 
# prediction algorithm
def get_sequences(data, seq_len):
    size = len(data) - seq_len
    dshape = data[0].shape
    seqs = np.zeros(shape=(size, seq_len, *dshape))
    results = np.zeros(shape=(size, *dshape))
    for i in range(size):
        seqs[i] = data[i:i+seq_len]
        results[i] = data[i+seq_len]
    return seqs, results
