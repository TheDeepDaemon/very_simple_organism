import pygame
import random
import math
import cv2
import numpy as np
from numpy.lib.function_base import angle
from pygame.transform import rotate
from scipy import ndimage
from tensorflow.core.protobuf.meta_graph_pb2 import AssetFileDef
import constants


def get_pixels(disp, startX, endX, startY, endY):
    window_pixel_matrix = pygame.surfarray.array3d(disp)
    
    # get pixels from a box on the screen
    return window_pixel_matrix[int(startX):int(endX), int(startY):int(endY), :]


def crop_subimage(image, size):
    middle_x = image.shape[0] / 2
    middle_y = image.shape[1] / 2
    halfw = size[0] / 2
    rotated_image = image[
        int(middle_x):int(middle_x + size[1]), 
        int(middle_y-halfw):int(middle_y+halfw), :]
    return rotated_image


def get_subimage(disp, angle, size, position):
    # 1.12 ≈ √(1.25)
    size_x = (size[0] * 1.12) + 1
    size_y = (size[1] * 1.12) + 1
    
    left_edge = position[0] - size_x
    right_edge = position[0] + size_x
    
    top_edge = position[1] - size_y
    bottom_edge = position[1] + size_y
    
    # sample from the section of the screen at [pos] of [size]
    image = get_pixels(
        disp,
        left_edge, right_edge, 
        top_edge, bottom_edge)
    
    middle_x = image.shape[0] / 2
    middle_y = image.shape[1] / 2
    halfw = size[0] / 2
    cropped = image[
        int(middle_x - halfw):int(middle_x + halfw), 
        int(middle_y - halfw):int(middle_y + halfw), :]
    
    angle = angle * (180.0 / math.pi)
    rotated_image = ndimage.rotate(image, -angle)
    
    rotated_image = crop_subimage(rotated_image, size)
    
    return rotated_image, cropped


def mouse_angle(pos):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    diry = mouse_y - pos[1]
    dirx = mouse_x - pos[0]
    return math.atan2(diry, dirx)


def draw_array(disp, arr, pos):
    surface = pygame.surfarray.make_surface(arr)
    disp.blit(surface, pos)


def rand_screen_pos(SCREEN_WIDTH, SCREEN_HEIGHT):
    return random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)


def draw_minidisplay(display, pixels, draw_border=True):
    x = constants.MINIDISPLAY_LOCATION
    y = 0
    constants.MINIDISPLAY_LOCATION += pixels.shape[1]
    if draw_border:
        pixels[:,0,:] = 0
        pixels[0,:,:] = 0
        pixels[:,-1,:] = 0
        pixels[-1,:,:] = 0
    draw_array(display, pixels * 255.0, (x, y))


def subimage_to_inputs(subimg):
    resized = cv2.resize(subimg, dsize=(16, 16), interpolation=cv2.INTER_AREA)
    img = to_greyscale(resized)
    return np.reshape(img, newshape=(16, 16, 1))


def to_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# convert 1d greyscale color channel
# to 3d color channel
def convert1dto3d(arr):
    new_arr = np.zeros(
        shape=(arr.shape[0], arr.shape[1], 3),
        dtype=np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i, j, 0] = arr[i, j, 0]
            new_arr[i, j, 1] = arr[i, j, 0]
            new_arr[i, j, 2] = arr[i, j, 0]
    return new_arr


# add_up x and y are to let you know how much to add to the position
def position_to_grid(pos, cell_size):
    x = int(math.ceil(pos[0] / cell_size))
    y = int(math.ceil(pos[1] / cell_size))
    add_up_x = int((x * cell_size) - pos[0])
    add_up_y = int((y * cell_size) - pos[1])
    return x, y, add_up_x, add_up_y


def add_tuple(tuple1, tuple2):
    return (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1])


def subtract_tuple(tuple1, tuple2):
    return (tuple1[0] - tuple2[0], tuple1[1] - tuple2[1])


def calc_view_position(pos, angle, view_size):
    relative_pos = (
        math.cos(angle) * view_size, 
        math.sin(angle) * view_size)
    return add_tuple(pos, relative_pos)

