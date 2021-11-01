import numpy as np

# compare patches of an image
def compare_patches(x1, x2):
    return np.sum(np.abs(x1 - x2))



# finds repeating patterns at similar locations
def compare_images(x1, x2, kernel_size, nearness=(1, 1)):
    for i in range(x1):
        for j in range(x2):
            pass

x1 = np.array([[1, 1], [1, 1]], dtype=np.float32)
x2 = np.array([[0, .9], [.9, 1]], dtype=np.float32)

print(compare_patches(x1, x2))
