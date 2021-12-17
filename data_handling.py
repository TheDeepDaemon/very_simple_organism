import cppfunctions
import numpy as np
from scipy import ndimage
import random
import math


# gives an index for an array of size n
def rand_index(n):
    return random.randint(0, n-1)


# create a row of ones and zeros,
# with a specific number of ones
def random_data_entry(size, num_ones):
    arr = np.zeros(shape=(size,), dtype=np.float32)
    ints = np.arange(size)
    np.random.shuffle(ints)
    rand_ints = ints[:num_ones]
    arr[rand_ints] = 1.0
    return arr


def generate_random_groups(size, num_groups, num_ones=2):
    groups = np.zeros(shape=(num_groups, size), dtype=np.float32)
    for i in range(num_groups):
        unique = False
        while not unique:
            groups[i] = random_data_entry(size, num_ones)
            
            unique = True
            for j in range(i):
                if np.array_equal(groups[i], groups[j]):
                    unique = False
    return groups


# generates 1 or 0 values
# in the shape of a matrix.
# used for testing and debugging.
def generate_random_data(rows, cols, num_groups, num_ones=2, variation=0.1):
    groups = generate_random_groups(cols, num_groups, num_ones)
    data = np.zeros(shape=(rows, cols), dtype=np.float32)
    group_inds = np.zeros(shape=(data.shape[0],), dtype=int)
    for i in range(data.shape[0]):
        r = rand_index(num_groups)
        group_inds[i] = r
        data[i] = groups[r]
    rand_arr = np.random.uniform(-1.0, 1.0, rows * cols)
    offset = np.reshape(rand_arr, newshape=(rows, cols)) * variation
    data += offset
    return data, groups, group_inds


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
def find_groups_old(data, split_size, markov_iterations, clustering_power, clustering_inflation):
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
def find_absolute_groups_old(data, stop_at, split_size, markov_iterations, clustering_power, clustering_inflation):
    cppfunctions.scale_up(data)
    groups = find_groups(data, split_size, markov_iterations, clustering_power, clustering_inflation)
    while len(groups) > stop_at:
        new_groups = find_groups(groups, split_size, markov_iterations, clustering_power, clustering_inflation)
        if len(new_groups) > 1:
            groups = new_groups
        else:
            break
    return cppfunctions.remove_similar(groups)


def augment_data(data, divisions=8, rows=None, cols=None):
    new_list = []
    for entry in data:
        roof = divisions + 1
        new_list.append(entry)
        for i in range(divisions):
            fraction = (i + 1) / roof
            angle = fraction * 90.0
            rot_img = ndimage.rotate(entry, angle, reshape=False)
            new_list.append(rot_img)
    new_list = np.array(new_list, dtype=np.float32)
    cppfunctions.apply_sigmoid(new_list)
    data = cppfunctions.augment_data_fr(new_list, rows, cols)
    return data


def select_data(data, num_to_extract):
    data = cppfunctions.remove_low_norms(data)
    sample_size = min(num_to_extract, len(data))
    sample = np.random.randint(0, len(data), size=sample_size)
    return data[sample]


def difference_between_groups(group1, group2):
    differences = np.zeros(shape=(group1.shape[0], group2.shape[0]), dtype=float)
    for i in range(group1.shape[0]):
        for j in range(group2.shape[0]):
            differences[i, j] = np.sum(np.abs(group1[i] - group2[j]))
    tot_size = differences.size * group1.size
    s = float(np.sum(differences))
    if tot_size is 0:
        tot_size = 1
    return s / float(tot_size)


def select_a_grouping(data, index, threshold):
    in_group = []
    compliment = []
    row_size = data[index].size
    for i in range(len(data)):
        difference = np.sum(np.abs(data[index] - data[i]))
        difference /= row_size
        if difference < threshold:
            in_group.append(i)
        else:
            compliment.append(i)
    return np.array(in_group, dtype=int), np.array(compliment, dtype=int)


def similarity_in_group(data):
    size = data.shape[0]
    diffs = 0.0
    for i in range(size):
        for j in range(size):
            d = np.sum(np.abs(data[i] - data[j]))
            diffs += d
    return diffs / float(size * size)


def eval_grouping(group, compliment):
    
    if len(group) != 0:
        between_groups = difference_between_groups(group, compliment)
        within_groups = similarity_in_group(group)
        if within_groups != 0:
            return between_groups / within_groups
        return math.inf
    return None


def get_a_grouping(data, num_examples, threshold):
    # ratings for each grouping on how well it represents the division
    ratings = np.zeros((num_examples,), dtype=float)
    groups = []
    for i in range(num_examples):
        # pick a random index
        r = rand_index(len(data))
        
        # evaluate how good it is
        group, compliment = select_a_grouping(data, r, threshold)
        rating = eval_grouping(group, compliment)
        ratings[i] = rating
        groups.append((group, compliment, rating))
    return groups[np.argmax(ratings)]

data, groups, indices = generate_random_data(10, 4, 2, 2, 0.05)

def split_group(data, granularity=10, num_examples=3):
    groups = []
    ratings = np.zeros((granularity,), dtype=float)
    for i in range(granularity):
        t = i / float(granularity)
        result = get_a_grouping(data, num_examples, threshold=t)
        _, _, rating = result
        groups.append(result)
        if rating is None:
            rating = 0
        ratings[i] = rating
    best = np.argmax(ratings)
    return groups[best]


# gets a number to tell whether to split the group up
def should_split(data, split_threshold):
    shuffled_data = np.copy(data)
    np.random.shuffle(shuffled_data)
    rows = data.shape[0]
    cols = data.shape[1]
    dist = 0
    if shuffled_data.size > 0:
        for i in range(cols):
            val = shuffled_data[0, i]
            for j in range(1, rows):
                data_entry = shuffled_data[j, i]
                dist += math.pow(data_entry - val, 2)
                val = data_entry
    return dist >= split_threshold


def contains_mixed_group(groups, split_threshold):
    for g in groups:
        if should_split(g, split_threshold):
            return True
    return False


def find_overlapping(groups, threshold):
    for i, g1 in enumerate(groups):
        for j, g2 in enumerate(groups):
            if i != j:
                combined = np.concatenate((g1, g2))
                if not should_split(combined, threshold):
                    groups = [grp for k, grp in enumerate(groups) if (k != i and k != j)]
                    groups.append(combined)
                    return groups


def rough_grouping_algorithm(data, clustering_iterations, inflation, split_threshold):
    if inflation < 3.0:
        return None
    
    indices = cppfunctions.find_groups(data, clustering_iterations, 2, inflation)
    
    found_groups = []
    for ind in indices:
        found_groups.append(data[ind])
    
    # if the group size is smaller than what it should be, this loop will fix that
    if contains_mixed_group(found_groups, split_threshold):
        while contains_mixed_group(found_groups, split_threshold):
            largest_ind = 0
            largest = 0
            for i, grp in enumerate(found_groups):
                spl = should_split(grp, split_threshold)
                if spl > largest:
                    largest_ind = i
                    largest = spl
            
            in_group, compliment, _ = split_group(found_groups[largest_ind])
            in_group = found_groups[largest_ind][in_group]
            compliment = found_groups[largest_ind][compliment]
            del found_groups[largest_ind]
            found_groups.append(in_group)
            found_groups.append(compliment)
    
    # if find similar groups, stick them together
    while True:
        ret = find_overlapping(found_groups, split_threshold)
        
        # if not none, then found_groups needs to be updated
        # else, end it because the groups are distinct
        if ret is not None:
            found_groups = ret
        else:
            break
    
    return found_groups

def grouping_algorithm(data, split_size, clustering_iterations, inflation, split_threshold):
    args = clustering_iterations, inflation, split_threshold
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
    
    for data_subset in split_data:
        clusters = rough_grouping_algorithm(data_subset, *args)
        for c in clusters:
            c = np.array(c, dtype=np.float32)
            average_example = cppfunctions.average_of_arrs(c)
            groups.append(average_example)
    
    return groups


def find_groups(data, split_size, clustering_iterations, inflation, split_threshold, stop_at=16):
    args = split_size, clustering_iterations, inflation, split_threshold
    if stop_at < 1:
        stop_at = 1
    
    for _ in range(100):
        groups = grouping_algorithm(data, *args)
        if len(groups) <= stop_at:
            return groups
        else:
            print("group size: ", len(groups))
            data = np.array(groups, dtype=np.float32)
        


# example:
if __name__ == "__main__":
    data, groups, group_inds = generate_random_data(200, 8, 4, 3, 0.25)
    print(len(groups))
    print("num groups: ", len(groups))
    for g in groups:
        print(np.round(g, decimals=3))
    
    
    groups = find_groups(data, 32, 100, 9.0, 1.0, stop_at=8)
    print("\nnum groups found: ", len(groups))
    for group in groups:
        print(np.round(group, decimals=2))
    print("num groups found: ", len(groups))
