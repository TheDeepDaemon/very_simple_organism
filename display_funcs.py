import pygame
import random
import math
import cv2
import numpy as np


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


def draw_minidisplay(display, pixels, x, y, draw_border=True):
    if draw_border:
        pixels[:,0,:] = 0
        pixels[0,:,:] = 0
        pixels[:,-1,:] = 0
        pixels[-1,:,:] = 0
    draw_array(display, pixels * 255.0, (x, y))


def subimage_to_inputs(subimg):
    resized = cv2.resize(subimg, dsize=(16, 16), interpolation=cv2.INTER_AREA)
    return resized


def outputs_to_image(outputs):
    img = np.zeros(
        shape=(outputs.shape[0], outputs.shape[1], 3), dtype=np.float32)
    for i in range(len(outputs)):
        for j in range(len(outputs[i])):
            img[i, j, 0] = outputs[i, j, 0]
            img[i, j, 1] = outputs[i, j, 0]
            img[i, j, 2] = outputs[i, j, 0]
    return img


def to_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert3dto1d(arr):
    new_arr = np.zeros(shape=(arr.shape[0], arr.shape[1], 3))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i, j, 0] = arr[i, j, 0]
            new_arr[i, j, 1] = arr[i, j, 0]
            new_arr[i, j, 2] = arr[i, j, 0]
    return new_arr
