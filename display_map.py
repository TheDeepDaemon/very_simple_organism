from constants import *
import numpy as np
import cv2
from display_util import *
import colors


def get_simulated_map(model, memory, center_x, center_y, rows, cols, cell_size):
    img = np.zeros(shape=(rows, cols), dtype=np.float32)
    
    inputs = []
    for i in range(rows):
        for j in range(cols):
            begin_i = i * cell_size
            begin_j = j * cell_size
            end_i = begin_i + cell_size
            end_j = begin_j + cell_size
            inputs.append(memory[begin_i:end_i, begin_j:end_j])
    
    outputs = model.predict(inputs)
    
    for i in range(rows):
        for j in range(cols):
            begin_i = i * cell_size
            begin_j = j * cell_size
            end_i = begin_i + cell_size
            end_j = begin_j + cell_size
            img[begin_i:end_i, begin_j:end_j] = outputs[i]
    
    return False



def display_map(display, real_map_pixels, simulated_map_pixels):
    # clear screen
    display.fill(colors.BLACK)
    
    half_screen_width = SCREEN_WIDTH / 2
    
    simulated_map_pixels *= 255
    
    img1 = cv2.resize(
        outputs_to_image(real_map_pixels), (half_screen_width, half_screen_width))
    img2 = cv2.resize(
        outputs_to_image(simulated_map_pixels), (half_screen_width, half_screen_width))
    
    left_edge = 0
    bottom_edge = SCREEN_HEIGHT / 4
    
    # draw actual pixels
    draw_array(display, img1, (left_edge, bottom_edge))
    
    # draw simulated
    draw_array(display, img2, (half_screen_width, bottom_edge))
