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
from game_object import GameObject
from maze import Maze
import image
import cv2
import matplotlib.pyplot as plt


AGENT_COLLISION_TYPE = 1
MAZE_COLLISION_TYPE = 2
GOAL_COLLISION_TYPE = 3


SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 60
AGENT_SPEED = 10000 * 40
AGENT_TURN_SPEED = 0.025
AGENT_VIEW_SHAPE = (128, 96)
MAZE_POSITION = (30, 30)
MAZE_SIZE = 500


def wrap(n, size):
    if n < 0:
        return size + n
    return n

def random_maze_edge(width, height):
    if random.randint(0, 1) == 0:
        x, y = ((np.random.randint(0, width) * 2) + 1), (random.randint(0, 1) * -1)
        xg, yg = (width - ((x - 1) / 2) - 1), wrap(-1 * (y + 1), height)
        return x, y, xg, yg
    else:
        x, y = (random.randint(0, 1) * -1), np.random.randint(0, height)
        xg, yg = wrap(-1 * (x + 1), width), (height - y - 1)
        return x, y, xg, yg


def grid_activation(x, size):
    if int(x / size) % 2 == 0:
        return 1.0
    return 0.0

def grid_activation_inv(x, size):
    if int(x / size) % 2 == 1:
        return 1.0
    return 0.0

def grid_position(x, y, grid_x, grid_y, cell_size, maze_size):
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




def func(display, image):
    draw_minidisplay(
        display, cv2.resize(cv2.rotate(convert3dto1d(image), rotateCode=cv2.ROTATE_90_CLOCKWISE), dsize=(SCREEN_WIDTH, SCREEN_HEIGHT)), 0, 0)


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
    
    def create_gameobject(self, x, y, width, height, color, collision_type=1, static=False):
        obj = GameObject(
            self, (x, y), 
            (width, height), 
            col_type=collision_type, 
            color=color, static=static)
        self.game_objects.append(obj)
        return obj
    
    def create_maze(self, maze_size, dim, x_pos=0, y_pos=0):
        edge_width = 4
        maze = Maze(dim, dim)
        maze.gen_random_maze()
        edges = maze.extract_maze_edges()
        
        cell_size = maze_size / dim
        cell_size_half = cell_size / 2
        self.maze_cell_size = cell_size
        self.maze_grid_size = dim
        
        rx, ry, xg, yg = random_maze_edge(dim, dim)
        edges[rx][ry] = False
        
        goal_x = x_pos + cell_size_half + (xg * cell_size)
        goal_y = y_pos + cell_size_half + (yg * cell_size)
        
        # place flag
        self.create_gameobject(
            goal_x, goal_y, cell_size / 2, cell_size / 2, 
            colors.MAGENTA, collision_type=GOAL_COLLISION_TYPE, static=True)
        
        for i_ in range(int(len(edges) / 2)):
            i = i_ * 2
            # draw left sides
            for j in range(len(edges[i])):
                if edges[i][j]:
                    x = i_ * cell_size
                    y = j * cell_size + cell_size_half
                    color = (0, float(j) * 255.0 / len(edges[i]), 1.0 * 255)
                    color = colors.WHITE
                    self.create_gameobject(x_pos + x, y_pos + y, edge_width, cell_size, color=color, static=True)
            
            # draw top and bottom sides
            for j in range(len(edges[i+1])):
                if edges[i+1][j]:
                    x = i_ * cell_size + cell_size_half
                    y = j * cell_size
                    color = (0, 1.0 * 255, float(i) * 255.0 / len(edges))
                    color = colors.WHITE
                    self.create_gameobject(x_pos + x, y_pos + y, cell_size, edge_width, color=color, static=True)
        
        # draw right sides
        for j in range(len(edges[-1])):
            if edges[-1][j]:
                x = maze_size
                y = j * cell_size + cell_size_half
                color = colors.CYAN
                color = colors.WHITE
                self.create_gameobject(
                    x_pos + x, y_pos + y, edge_width, cell_size, 
                    color, static=True)
    
    
    def run_game(self):
        agent = Agent(self, self.starting_pos, 12, AGENT_COLLISION_TYPE)
        self.agent = agent
        self.game_objects.append(agent)
        forward = False
        right = False
        left = False
        self.create_maze(maze_size=MAZE_SIZE, dim=10, x_pos=MAZE_POSITION[0], y_pos=MAZE_POSITION[1])
        
        handler = self.space.add_collision_handler(AGENT_COLLISION_TYPE, GOAL_COLLISION_TYPE)
        handler.begin = self.reset_player_position
        
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
                        x, y = agent.body.position
                        print(grid_position(
                            x, y, MAZE_POSITION[0], MAZE_POSITION[1], 
                            self.maze_cell_size, MAZE_SIZE))
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
            
            self.display.fill(colors.BLACK)
            [obj.draw() for obj in self.game_objects]
            
            try:
                subimage = image.get_subimage(
                    self.display, -agent.body.angle, 
                    AGENT_VIEW_SHAPE, (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
                subimage = np.array(subimage, dtype=np.float32) / 255.0
                x = np.reshape(to_greyscale(subimage_to_inputs(subimage)), newshape=(16, 16, 1))
                
                draw_minidisplay(self.display, subimage, 0, 0)
                #draw_minidisplay(self.display, x, 0, 128)
                
                
                output = agent.brain.forward_process(x)
                if output is not None:
                    groups = output
                    print("groups: ", len(groups))
                    grps = groups[:10]
                    for group in grps:
                        plt.imshow(np.reshape(group, newshape=(4, 4, 1)))
                        plt.show()
                
                #y = agent.brain.pred_next_frame()
                #y = cv2.resize(y, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
                #draw_minidisplay(self.display, y, 128, 0)
            except Exception as e:
                print(e)
            
            
            if left:
                agent.body.angle += math.pi * AGENT_TURN_SPEED
            if right:
                agent.body.angle -= math.pi * AGENT_TURN_SPEED
            
            
            if forward:
                agent.body.apply_impulse_at_local_point((AGENT_SPEED, 0), (0, 0))
            
            pygame.display.update()
            self.clock.tick(FPS)
            self.space.step(1.0 / FPS)



pygame.init()
display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
space = pymunk.Space()

system = Game(display, space, clock)

system.run_game()
pygame.quit()

