import os
from types import new_class
from PIL.Image import new
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pygame
import pymunk
from agent import Agent
import numpy as np
from display_funcs import *
import colors
from maze import create_maze
import image
import cv2
import matplotlib.pyplot as plt
from create_gameobject import create_gameobject


AGENT_COLLISION_TYPE = 1
MAZE_COLLISION_TYPE = 2
GOAL_COLLISION_TYPE = 3

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 60
AGENT_SPEED = 10000 * 40
AGENT_TURN_SPEED = 0.025
AGENT_VIEW_SHAPE = (128, 128)
MAZE_POSITION = (30, 30)
MAZE_SIZE = 500

RAW_MINIDISPLAY = False


def grid_activation(x, size):
    if int(x / size) % 2 == 0:
        return 1.0
    return 0.0

def grid_activation_inv(x, size):
    if int(x / size) % 2 == 1:
        return 1.0
    return 0.0

def grid_position(pos, grid_x, grid_y, cell_size, maze_size):
    x, y = pos
    x_coord = x - grid_x
    y_coord = y - grid_y
    nodes = np.array([
        grid_activation(x_coord, cell_size), 
        grid_activation(y_coord, cell_size),
        grid_activation(x_coord, maze_size / 8), 
        grid_activation(y_coord, maze_size / 8),
        grid_activation(x_coord, maze_size / 4), 
        grid_activation(y_coord, maze_size / 4),
        grid_activation(x_coord, maze_size / 2), 
        grid_activation(y_coord, maze_size / 2),
        grid_activation_inv(x_coord, cell_size), 
        grid_activation_inv(y_coord, cell_size),
        grid_activation_inv(x_coord, maze_size / 8), 
        grid_activation_inv(y_coord, maze_size / 8),
        grid_activation_inv(x_coord, maze_size / 4), 
        grid_activation_inv(y_coord, maze_size / 4),
        grid_activation_inv(x_coord, maze_size / 2), 
        grid_activation_inv(y_coord, maze_size / 2),], dtype=np.float32)
    return nodes


class Game:
    
    def __init__(self, display, space, clock):
        self.camera_pos = (0, 0)
        self.game_objects = []
        self.display = display
        self.space = space
        self.clock = clock
        space.damping = 0.01
        self.starting_pos = (50, 550)
    
    
    def reset_player_position(self, arbiter, space, data):
        self.agent.body.position = self.starting_pos
        self.agent.body.velocity = (0, 0)
        return True
    
    
    def create_and_add_gameobject(self, x, y, width, height, color, collision_type=1, static=False):
        return create_gameobject(self, x, y, width, height, color, collision_type, static)
    
    
    def run_game(self):
        
        # init variables
        agent = Agent(self, self.starting_pos, 12, AGENT_COLLISION_TYPE)
        self.agent = agent
        self.game_objects.append(agent)
        pos_prev = agent.body.position
        forward = False
        right = False
        left = False
        create_maze(self, maze_size=MAZE_SIZE, dim=10, x_pos=MAZE_POSITION[0], y_pos=MAZE_POSITION[1])
        
        # set the flag to send the agent/player to the beginning
        handler = self.space.add_collision_handler(AGENT_COLLISION_TYPE, GOAL_COLLISION_TYPE)
        handler.begin = self.reset_player_position
        
        # main loop
        while True:
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
                    elif event.key == pygame.K_q:
                        print()
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:
                        forward = False
                    elif event.key == pygame.K_RIGHT:
                        right = False
                    elif event.key == pygame.K_LEFT:
                        left = False
            
            
            # set camera position
            cam_x = agent.body.position[0] - (SCREEN_WIDTH / 2)
            cam_y = agent.body.position[1] - (SCREEN_HEIGHT / 2)
            self.camera_pos = (cam_x, cam_y)
            
            # set the screen to black, then draw everything
            self.display.fill(colors.BLACK)
            [obj.draw() for obj in self.game_objects]
            
            # do all the AI stuff
            try:
                subimage = image.get_subimage(
                    self.display, -agent.body.angle, 
                    AGENT_VIEW_SHAPE, (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
                subimage = np.array(subimage, dtype=np.float32) / 255.0
                x = np.reshape(to_greyscale(subimage_to_inputs(subimage)), newshape=(16, 16, 1))
                
                # raw means the literal image patch that is used
                # else means show the processed (shrunk) image
                if RAW_MINIDISPLAY:
                    draw_minidisplay(self.display, subimage, 0, 0)
                else:
                    draw_minidisplay(
                        self.display, cv2.resize(outputs_to_image(x), (128, 128)), 0, 0)
                
                # grid pos is shape (16,)
                grid_pos = grid_position(
                    agent.body.position, MAZE_POSITION[0], MAZE_POSITION[1], 
                    self.maze_cell_size, MAZE_SIZE)
                
                # pos delta is the change in position
                pos_delta = agent.body.position - pos_prev
                
                # feed input data
                agent.brain.process_inputs(
                    x, grid_activations=grid_pos, pos_delta=pos_delta)
                
                
                # create minidisplay for the nn model contents
                y = agent.brain.reconstruct_internal_model(x)
                if y is not None:
                    y = outputs_to_image(y)
                    y = cv2.resize(y, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
                    draw_minidisplay(self.display, y, 128, 0)
                
            except Exception as e:
                print(e)
            
            
            if left:
                agent.body.angle += math.pi * AGENT_TURN_SPEED
            if right:
                agent.body.angle -= math.pi * AGENT_TURN_SPEED
            
            
            if forward:
                agent.body.apply_impulse_at_local_point((AGENT_SPEED, 0), (0, 0))
            
            pos_prev = agent.body.position
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

