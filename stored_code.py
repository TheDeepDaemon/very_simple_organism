import pygame
import image
import math
import numpy as np
from display_funcs import *
import colors


SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 60
AGENT_SPEED = 400
AGENT_VIEW_SHAPE = (128, 128)

def recombination_procedure():
    for _ in range(0):
        pass # create new based on one point combination
    for _ in range(0):
        pass # two point combination
    for _ in range(0):
        pass # random weight selection
    for _ in range(0):
        pass # mutate
    # concatenate lists
    # selection process
    # sort based on fitness


def display_basic_env(display, agent):
    display.fill(colors.WHITE)
    pygame.draw.circle(
        display, colors.BLUE, 
        (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2), 100)
    agent.draw(display, agent.body.angle)

