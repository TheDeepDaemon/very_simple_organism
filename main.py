import os
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


def draw_agent_map(display, agent):
    map_img = agent.brain.read_map()
    
    if map_img is not None:
        img = cv2.resize(
            convert1dto3d(map_img), 
            (128, 128))
        
        draw_minidisplay(display, img, False)



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
        #self.game_objects.append(agent)
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
                        draw_agent_map(self.display, self.agent)
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
                    static_img = subimage_to_inputs(static_img)
                    
                    # raw means the literal image patch that is used
                    # else means show the processed (shrunk) image
                    if RAW_MINIDISPLAY:
                        draw_minidisplay(
                            self.display, subimage)
                    else:
                        draw_minidisplay(
                            self.display, 
                            cv2.resize(convert1dto3d(x), (128, 128)))
                    
                    map_input = self.agent.brain.map_input
                    if map_input is not None:
                        map_img = np.reshape(cv2.resize(map_input, (128, 128)), newshape=(128, 128, 1))
                        #draw_minidisplay(self.display, convert1dto3d(map_img))
                    
                    #if agent.brain.decoder is not None:
                        #draw_agent_map(self.display, self.agent)
                    
                    # pos delta is the change in position
                    pos_delta = subtract_tuple(agent_pos_prev, agent.body.position)
                    
                    # feed input data
                    agent.brain.process_inputs(x, static_img, pos_delta)
                    
                    # create minidisplay for the nn model contents
                    y = agent.brain.reconstruct_internal_model(x)
                    if y is not None:
                        y = convert1dto3d(y)
                        y = cv2.resize(
                            y, dsize=(128, 128), 
                            interpolation=cv2.INTER_LINEAR)
                        draw_minidisplay(self.display, y)
                    
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

