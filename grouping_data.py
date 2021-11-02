from typing_extensions import final
from numpy.testing._private.utils import _assert_valid_refcount
import cppfunctions
import numpy as np


# remove groups that have values further from 1 or 0
def remove_high_entropy(groups):
    entropy = cppfunctions.calc_groups_entropy(groups)
    ave_entropy = np.average(entropy)
    groups_to_keep = []
    for i, group in enumerate(groups):
        if entropy[i] < ave_entropy:
            groups_to_keep.append(group)
    return np.array(groups_to_keep, dtype=np.float32)


# average the contents of a list of groups
def average_of_groups(data, index_lists):
    ave_groups = []
    for index_list in index_lists:
        ave = np.zeros_like(data[0], dtype=np.float32)
        for i in index_list:
            ave += data[i]
        ave_groups.append(ave / float(len(index_list)))
    return ave_groups


# use clustering techinques to find groups of similar patterns
def find_groups(data, split_size, markov_iterations, clustering_power, clustering_inflation):
    num_splits = int(len(data) / split_size)
    
    np.random.shuffle(data)
    
    split_data = []
    for i in range(num_splits):
        begin = i * split_size
        end = begin + split_size
        split_data.append(data[begin:end])
    
    if split_size * num_splits < len(data):
        split_data.append(data[split_size * num_splits:])
    
    groups = []
    for sample in split_data:
        indices = cppfunctions.find_groups(sample, markov_iterations, clustering_power, clustering_inflation)
        clusters = []
        for ind in indices:
            clusters.append(data[ind])
        
        for cluster in clusters:
            group = cppfunctions.remove_outliers(cluster)
            average_example = cppfunctions.average_of_arrs(group)
            groups.append(average_example)
    
    return remove_high_entropy(groups)


# keeps grouping until it gets smaller than a certain size
def find_absolute_groups(data, stop_at, split_size, markov_iterations, clustering_power, clustering_inflation):
    cppfunctions.scale_up(data)
    groups = find_groups(data, split_size, markov_iterations, clustering_power, clustering_inflation)
    while len(groups) > stop_at:
        new_groups = find_groups(groups, split_size, markov_iterations, clustering_power, clustering_inflation)
        if len(new_groups) > 1:
            groups = new_groups
        else:
            break
    return cppfunctions.join_similar(groups)



# example:
if __name__ == "__main__":
    data = cppfunctions.generate_random_data(350000, 16)
    groups = find_absolute_groups(data, 90, 120, 100, 2, 100)
    print("groups: ", len(groups))
    for g in groups:
        print(np.round(g, decimals=3))

