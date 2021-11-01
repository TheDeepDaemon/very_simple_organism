import pygame
import random
import math
import cv2


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
