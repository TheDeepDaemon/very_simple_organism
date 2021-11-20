import numpy as np
from numpy.core.defchararray import index


class MemoryBuffer:
    
    def __init__(self, capacity, short_term_memsize, input_shape):
        shape = (capacity, *input_shape)
        self.memory = np.zeros(shape=shape, dtype=np.float32)
        self.capacity = capacity
        self.short_term_memsize = short_term_memsize
        self.index = 0
    
    def insert_memory(self, inputs):
        self.memory[self.index] = inputs
        self.index += 1
        self.index = self.index % self.capacity
        hit_end = (self.index == 0)
        return hit_end
    
    def mems_at(self, index):
        return self.memory[index-self.short_term_memsize:index]
    
    def recent_mems(self):
        return self.mems_at(self.index)
    
    def clear(self):
        self.memory.fill(0)
    
    def equals_last(self, inputs):
        return np.array_equal(self.memory[self.index-1], inputs)


