import numpy as np


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



def replace_kernel(weights, kernel, filter_index, channel_index=None):
    for i in range(len(kernel)):
        for j in range(len(kernel[i])):
            if channel_index is None:
                for c in range(len(weights[0][i][j])):
                    weights[0][i][j][c][filter_index] = kernel[i][j]
            else:
                weights[0][i][j][channel_index][filter_index] = kernel[i][j]


def set_layer_weights(model, kernels, layer_name):
    for layer in model.layers:
        weights = layer.get_weights()
        if layer.name == layer_name:
            for i, kernel in enumerate(kernels):
                replace_kernel(weights, kernel, i)
            layer.set_weights(weights)
            break

