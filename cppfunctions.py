# python file made to give access to C++ code
# the functions are an interface for the C++ functions
import ctypes
from ctypes import CDLL
import numpy as np


# easy reference to ctypes
cfloat = ctypes.c_float
cint64 = ctypes.c_int64
cdouble = ctypes.c_double





cpp_code_dir = '..\\bin\\'
fname = 'main.so'
path = cpp_code_dir + '\\' + fname
cpp_functions = CDLL(path)
def get_nparr_ptr(np_arr):
    c_type = None
    if np_arr.dtype == np.float32:
        c_type = cfloat
    elif np_arr.dtype == np.float64:
        c_type = cdouble
    elif np_arr.dtype == np.int32:
        c_type = ctypes.c_int32
    elif np_arr.dtype == np.int64:
        c_type = cint64
    elif np_arr.dtype == bool:
        c_type = ctypes.c_bool
    else:
        return None
    return np_arr.ctypes.data_as(ctypes.POINTER(c_type))


# takes an np array and outputs a ptr to the data, 
# and a series of ints that store the shape
# outputs as a tuple
def get_nparr_args(np_arr):
    shape = tuple([cint64(x) for x in np_arr.shape])
    return (get_nparr_ptr(np_arr=np_arr), *shape)


# take patches of an image (as a np array) 
# and convert it into data that can be processed
def images_to_matrix(imgs, wCols, wRows):
    jn = imgs.shape[1] - wCols + 1
    kn = imgs.shape[2] - wRows + 1
    row_size = wCols * wRows
    mat = np.zeros(shape=(imgs.shape[0] * jn * kn, row_size), dtype=np.float32)
    if len(imgs.shape) == 3: # make sure it has a valid shape
        args = (get_nparr_ptr(mat), *get_nparr_args(imgs), wCols, wRows)
        cpp_functions.imagesToMatrix(*args)
    return mat


# turns the rgb or rgba values in an image (as np array) and converts into a single row
def flatten_4th_dim(np_arr):
    shape = (np_arr.shape[0], np_arr.shape[1], np_arr.shape[2] * np_arr.shape[3])
    return np.reshape(np_arr, newshape=shape)


# apply the function to an image that has color channels as well
def images_to_matrix_4d(imgs, wCols, wRows):
    if len(imgs.shape) == 4:
        channels = imgs.shape[3]
        imgs = flatten_4th_dim(imgs)
        return images_to_matrix(imgs, wCols, wRows * channels)


# used to convert the contents of
# the returned values of find_groups
# to a python list that contains groups
def get_indices_as_list(indices):
    index_list = []
    for i in range(indices.shape[0]):
        if indices[i, 0] == -1:
            break
        group = []
        for j in range(indices.shape[1]):
            ind = indices[i, j]
            if ind == -1:
                break
            group.append(ind)
        index_list.append(group)
    return index_list


# apply a sigmoid to all elements of an array
def apply_sigmoid(np_arr):
    args = get_nparr_ptr(np_arr), ctypes.c_uint64(np_arr.size)
    cpp_functions.applySigmoid(*args)


# find the groups in the data by converting the relationships
# in the data to a markov matrix and multiplying by itself
# repeatedly in a clustering algorithm
def find_groups(data, iterations, power, inflation, max_num_groups=None):
    # the sigmoid makes the differences between the elements
    # of a vector more distinct (over or under the middle val)
    apply_sigmoid(data)
    rows = data.shape[0]
    if max_num_groups is None:
        max_num_groups = rows
    
    # the indices are initted to -1, so any vals that are -1
    # indicate that that has not been set
    indices = np.ones(shape=(max_num_groups, rows), dtype=np.int32) * -1
    cpp_functions.findGroups(
        *get_nparr_args(data), 
        get_nparr_ptr(indices), max_num_groups,
        iterations,
        cdouble(power), cdouble(inflation))
    return get_indices_as_list(indices)


# remove any entries that are sufficiently different
# from the existing group, uses second difference,
# so there are no hyperparameters needed
def remove_outliers(data):
    to_keep = np.zeros(data.shape[0], dtype=bool)
    cpp_functions.removeOutliers(
        get_nparr_ptr(data), 
        cint64(data.shape[0]), 
        cint64(data.shape[1]), 
        get_nparr_ptr(to_keep))
    new_data = []
    for i in range(len(data)):
        if to_keep[i]:
            new_data.append(data[i])
    return np.array(new_data, dtype=np.float32)


# remove entries that are too similar to others
def remove_similar(data):
    rows = data.shape[0]
    keep = np.zeros(rows, dtype=bool)
    cpp_functions.removeSimilar(
        *get_nparr_args(data), 
        get_nparr_ptr(keep), 
        cfloat(0.1))
    new_data = []
    for i in range(rows):
        if keep[i]:
            new_data.append(data[i])
    return new_data


# scales the data so the largest value is 1.0
def scale_up(arr):
    cpp_functions.scaleUp(get_nparr_ptr(arr), cint64(arr.size))


# average multiple arrays
# return the average of each in a 1d array
def average_of_arrs(arrs):
    ave = np.zeros_like(arrs[0], dtype=np.float32)
    n = arrs.shape[0]
    arr_size = int(arrs.size / n)
    args = get_nparr_ptr(arrs), cint64(n), cint64(arr_size), get_nparr_ptr(ave)
    cpp_functions.averageArr(*args)
    return ave


# get what the dot product of two vectors would be if they were normalized
# tells how similar they are in the direction they point
cpp_functions.normalizedDotProd.restype = cfloat
def normalized_dot(arr1, arr2):
    args = get_nparr_ptr(arr1), get_nparr_ptr(arr2), cint64(3)
    return cpp_functions.normalizedDotProd(*args)


# calculate the entropy of the vector,
# a measure of how many vector values
# are close to 0 or 1
cpp_functions.calcEntropy.restype = cfloat
def calc_entropy(arr):
    args = get_nparr_ptr(arr), cint64(arr.size)
    return cpp_functions.calcEntropy(*args)


# takes a list of np arrays, returns the entropy of each
def calc_groups_entropy(arrs):
    entropy = np.zeros(len(arrs), dtype=np.float32)
    for i, arr in enumerate(arrs):
        entropy[i] = calc_entropy(arr)
    return entropy


# augment data by adding entries 
# that are flipped and rotated (fr)
def augment_data_fr(data, rows=None, cols=None):
    if rows is None:
        rows = data.shape[1]
    if cols is None:
        cols = data.shape[2]
    
    if data.size == data.shape[0] * rows * cols:
        
        n_augmented = None
        if rows == cols:
            n_augmented = data.shape[0] * 6
        else:
            n_augmented = data.shape[0] * 4
        
        augmented = np.zeros(shape=(n_augmented, rows, cols), dtype=np.float32)
        
        cpp_functions.augmentData(
            get_nparr_ptr(data), cint64(data.shape[0]), 
            cint64(rows), cint64(cols), 
            get_nparr_ptr(augmented))
        return augmented
    else:
        print("Input for augmenting data is wrong shape.")


# takes patterns found in the data and outputs
# the values to set the weights to
def create_weights(data):
    weights = np.copy(data)
    vecSize = cint64(int(weights.size / weights.shape[0]))
    args = get_nparr_ptr(weights), cint64(weights.shape[0]), vecSize
    cpp_functions.toWeights(*args)
    return weights


cpp_functions.findCutoff.restype = cint64
def remove_low_norms(data):
    # sort based on norm to remove blank data, 
    # images that are all or mostly black will have low norms.
    norms = np.linalg.norm(data, axis=1)
    indices = np.argsort(norms)
    sorted_norms = norms[indices]
    args = get_nparr_ptr(sorted_norms), cint64(sorted_norms.size)
    cutoff = cpp_functions.findCutoff(*args)
    print("cutoff: ", cutoff / float(len(data)))
    return data[indices[cutoff:]]

