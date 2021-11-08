
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
