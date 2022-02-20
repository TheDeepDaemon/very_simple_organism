from dis import dis
import os
from tkinter import HORIZONTAL
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # turns off access to GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from types import new_class
from PIL.Image import new
from numpy.core.fromnumeric import size
from numpy.lib.type_check import real
import pygame
import pymunk
from agent import Agent
import numpy as np
from display_util import *
import colors
from maze import create_maze
import cv2
import matplotlib.pyplot as plt
from create_gameobject import create_gameobject
from constants import *
import math


def display_raw(display, subimage, x):
    # 'raw' means the literal image patch that is used
    # else means show the processed (shrunk) image
    if RAW_MINIDISPLAY:
        draw_minidisplay(display, subimage)
    else:
        draw_minidisplay(
            display, 
            cv2.resize(convert1dto3d(x), (128, 128)))


def display_reconstructed(display, brain, x):
    y = brain.reconstruct_internal_model(x)
    if y is not None:
        y = convert1dto3d(y)
        y = cv2.resize(
            y, dsize=(128, 128), 
            interpolation=cv2.INTER_LINEAR)
        draw_minidisplay(display, y)


def display_map_input(display, brain):
    map_input = brain.map_input
    if map_input is not None:
        map_img = np.reshape(cv2.resize(map_input, (128, 128)), newshape=(128, 128, 1))
        draw_minidisplay(display, map_img)


def display_map_here(display, brain):
    if brain.decoder is not None:
        map_img = brain.read_map()
        #map_img = brain.read_latent()
        if map_img is not None:
            img = cv2.resize(map_img, (128, 128))
            img = np.reshape(img, newshape=(128, 128, 1))
            draw_minidisplay(display, img)


def reconstruct_map(display, brain):
    if (brain.decoder is not None) and (brain.internal_map is not None):
        pos_x, pos_y = brain.get_grid_location()
        grid_cells = 6
        bmap = brain.internal_map[
            pos_x-grid_cells:pos_x+grid_cells, 
            pos_y-grid_cells:pos_y+grid_cells]
        
        images_list = []
        
        n1 = bmap.shape[0]
        n2 = bmap.shape[1]
        
        for i in range(n1):
            for j in range(n2):
                images_list.append(bmap[i,j])
        
        images_list = np.array(images_list, dtype=np.float32)
        
        images_list = brain.decoder.predict(images_list)
        
        img_shape = images_list.shape[1:]
        decoded_images = np.zeros(
            shape=(n1, n2, *img_shape), 
            dtype=np.float32)
        
        k = 0
        for i in range(n1):
            for j in range(n2):
                decoded_images[i, j] = images_list[k]
                k += 1
        
        single_img_shape = decoded_images.shape[2:]
        width = decoded_images.shape[0] * single_img_shape[0]
        height = decoded_images.shape[1] * single_img_shape[1]
        the_rest = single_img_shape[2:]
        image_shape = width, height, *the_rest
        output_image = np.zeros(shape=image_shape, dtype=np.float32)
        
        n3 = single_img_shape[0]
        n4 = single_img_shape[1]
        
        for i in range(n1):
            for j in range(n2):
                for k in range(n3):
                    for l in range(n4):
                        grid_x = (i * n3) + k
                        grid_y = (j * n4) + l
                        output_image[grid_x, grid_y] = decoded_images[i, n2 - j - 1, k, l]
        
        size = 128
        img = cv2.resize(output_image, (size, size))
        img = np.reshape(img, newshape=(size, size, 1))
        
        draw_minidisplay(display, img)


def show_cell(display, brain):
    x, y = brain.position
    min_x = int(math.floor(x / 64)) * 64
    min_y = int(math.floor(y / 64)) * 64
    
    # calculate position relative to the agent
    relative_x = min_x - x
    relative_y = min_y - y
    
    # get the position of the agent on the screen
    center_x = SCREEN_WIDTH / 2
    center_y = SCREEN_HEIGHT / 2
    
    # get the cell corner on the screen
    min_x = center_x + relative_x
    min_y = center_y + relative_y
    
    # get the opposite corner
    max_x = min_x + 64
    max_y = min_y + 64
    
    min_y = SCREEN_WIDTH - min_y
    max_y = SCREEN_WIDTH - max_y
    
    edge_color = colors.CRIMSON
    pygame.draw.line(display, edge_color, (min_x, min_y), (max_x, min_y))
    pygame.draw.line(display, edge_color, (min_x, min_y), (min_x, max_y))
    pygame.draw.line(display, edge_color, (max_x, min_y), (max_x, max_y))
    pygame.draw.line(display, edge_color, (min_x, max_y), (max_x, max_y))



class Game:
    
    def __init__(self, display, space, clock):
        self.camera_pos = (0, 0)
        self.game_objects = []
        self.display = display
        self.space = space
        self.clock = clock
        space.damping = 0.01
        self.starting_pos = (50, 550)
    
    
    def create_and_add_gameobject(self, x, y, width, height, color, collision_type=1, static=False):
        return create_gameobject(self, x, y, width, height, color, collision_type, static)
    
    
    def run_game(self):
        
        # init variables
        agent = Agent(self, self.starting_pos, 12, AGENT_COLLISION_TYPE)
        self.agent = agent
        agent_pos_prev = agent.body.position
        
        forward = False
        right = False
        left = False
        create_maze(self, maze_size=MAZE_SIZE, dim=10, x_pos=MAZE_POSITION[0], y_pos=MAZE_POSITION[1])
        
        paused = False
        
        # main loop
        while True:
            # reset minidisplays
            constants.MINIDISPLAY_LOCATION = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        forward = True
                    elif event.key == pygame.K_RIGHT:
                        right = True
                    elif event.key == pygame.K_LEFT:
                        left = True
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_o:
                        reconstruct_map(self.display, self.agent.brain)
                        pygame.display.update()
                        paused = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:
                        forward = False
                    elif event.key == pygame.K_RIGHT:
                        right = False
                    elif event.key == pygame.K_LEFT:
                        left = False
            
            if not paused:
                
                # set camera position
                cam_x = agent.body.position[0] - (SCREEN_WIDTH / 2)
                cam_y = agent.body.position[1] - (SCREEN_HEIGHT / 2)
                self.camera_pos = (cam_x, cam_y)
                
                # set the screen to black, then draw everything
                self.display.fill(colors.BLACK)
                [obj.draw() for obj in self.game_objects]
                
                # do all the AI stuff in this try block
                try:
                    subimage, static_img = get_subimage(
                        self.display, -agent.body.angle, 
                        AGENT_VIEW_SHAPE, (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
                    subimage = np.array(subimage, dtype=np.float32) / 255.0
                    static_img = np.array(static_img, dtype=np.float32) / 255.0
                    
                    # convert to the 16x16x1 input shape
                    x = subimage_to_inputs(subimage)
                    
                    # draw raw inputs
                    display_raw(self.display, subimage, x)
                    
                    # create minidisplay for the nn model contents
                    display_reconstructed(display, self.agent.brain, x)
                    
                    # display the input to the agent map
                    display_map_input(self.display, self.agent.brain)
                    
                    # show what is decoded from the internal map at the current location
                    #display_map_here(self.display, self.agent.brain)
                    
                    # show reconstructed whole map
                    #reconstruct_map(self.display, self.agent.brain)
                    
                    # show which cell the agent is in
                    show_cell(self.display, self.agent.brain)
                    
                    # pos delta is the change in position
                    pos_delta = subtract_tuple(self.agent.body.position, agent_pos_prev)
                    
                    # feed input data
                    agent.brain.process_inputs(x, static_img, pos_delta)
                    
                except Exception as e:
                    print(e)
                
                if left:
                    agent.body.angle += math.pi * AGENT_TURN_SPEED
                if right:
                    agent.body.angle -= math.pi * AGENT_TURN_SPEED
                
                if forward:
                    agent.body.apply_impulse_at_local_point((AGENT_SPEED, 0), (0, 0))
                
                
                self.agent.draw()
                agent_pos_prev = agent.body.position
                pygame.display.update()
                self.clock.tick(FPS)
                self.space.step(1.0 / FPS)



if __name__ == "__main__":
    pygame.init()
    display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    space = pymunk.Space()

    system = Game(display, space, clock)

    system.run_game()
    pygame.quit()

