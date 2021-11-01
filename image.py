from numpy.lib.function_base import angle
import pygame
import math
import numpy as np
from pygame.transform import rotate
from scipy import ndimage


def get_pixels(disp, startX, endX, startY, endY):
    window_pixel_matrix = pygame.surfarray.array3d(disp)
    
    # get pixels from a box on the screen
    return window_pixel_matrix[int(startX):int(endX), int(startY):int(endY), :]


def get_subimage(disp, angle, size, position):
    # 1.12 ≈ √(1.25)
    size_x = (size[0] * 1.12) + 1
    size_y = (size[1] * 1.12) + 1
    
    left_edge = position[0] - size_x
    right_edge = position[0] + size_x
    
    top_edge = position[1] - size_y
    bottom_edge = position[1] + size_y
    
    # sample from the section of the screen at [pos] of [size]
    pixels = get_pixels(disp,
        left_edge, right_edge, 
        top_edge, bottom_edge)
    
    angle = angle * (180.0 / math.pi)
    rotated_image = ndimage.rotate(pixels, -angle)
    middle_x = rotated_image.shape[0] / 2
    middle_y = rotated_image.shape[1] / 2
    
    halfw = size[0] / 2
    image = rotated_image[
        int(middle_x):int(middle_x+size[1]), 
        int(middle_y-halfw):int(middle_y+halfw), :]
    
    return image

