from constants import *
import math



def cell_edges(agent_pos, cell_size):
    x = agent_pos[0]
    y = agent_pos[1]
    
    scaled_x = x / cell_size
    scaled_y = y / cell_size
    
    xmin = (math.floor(scaled_x))
    ymin = (math.floor(scaled_y))
    xmax = (math.floor(scaled_x) + 1)
    ymax = (math.floor(scaled_y) + 1)
    
    xmin = (xmin * cell_size)
    ymin = (ymin * cell_size)
    xmax = (xmax * cell_size)
    ymax = (ymax * cell_size)
    
    return xmin, xmax, ymin, ymax


def convert_to_view(xmin, xmax, ymin, ymax, agent_pos):
    
    x = agent_pos[0]
    y = agent_pos[1]
    
    # calculate position relative to agent
    xmin = xmin - x
    ymin = ymin - y
    xmax = xmax - x
    ymax = ymax - y
    
    # use position relative to agent to calculate screen position
    middle_x = SCREEN_WIDTH / 2
    middle_y = SCREEN_HEIGHT / 2
    xmin = middle_x + xmin
    ymin = middle_y + ymin
    xmax = middle_x + xmax
    ymax = middle_y + ymax
    
    ymin = SCREEN_HEIGHT - ymin
    ymax = SCREEN_HEIGHT - ymax
    
    return xmin, xmax, ymin, ymax


def map_edges(agent_pos, wsize, cell_size):
    
    xmin, xmax, ymin, ymax = cell_edges(agent_pos, cell_size)
    
    xmin, xmax, ymin, ymax = convert_to_view(xmin, xmax, ymin, ymax, agent_pos)
    
    mid_y = SCREEN_HEIGHT / 2
    mid_x = SCREEN_WIDTH / 2
    
    xmin = (wsize[0] / 2) + (xmin - mid_x)
    xmax = (wsize[0] / 2) + (xmax - mid_x)
    ymin = (wsize[1] / 2) + (ymin - mid_y)
    ymax = (wsize[1] / 2) + (ymax - mid_y)
    
    return int(xmin), int(xmax), int(ymin), int(ymax)

