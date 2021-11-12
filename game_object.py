from cv2 import circle
from numpy.core.fromnumeric import shape, size
import pygame
from pygame import display
import pymunk
import colors
import math
from util import *


# functions for creating pymunk physics objects
def create_box(body, params):
    return pymunk.Poly.create_box(body, params)

def create_triangle(body, params):
    rad = params
    return pymunk.Poly(body, get_triangle_vertices(rad, 0))

def create_circle(body, radius):
    return pymunk.Circle(body, radius=radius)


# game object definition
class GameObject:
    
    def __init__(self, system, pos, size, col_type, color, static=False, shape_type='square', display_type=None):
        if display_type is None:
            display_type = shape_type
        shape_constructor = None
        
        if shape_type == 'square':
            shape_constructor=create_box
            params = make_square_tuple(size)
        elif shape_type == 'triangle':
            shape_constructor=create_triangle
            params = size
        elif shape_type == 'circle':
            shape_constructor=create_circle
            params = size
        
        
        if display_type == 'square':
            self.disp_size = make_square_tuple(size)
        elif display_type == 'triangle':
            self.disp_size = size
        elif display_type == 'circle':
            self.disp_size = size
        
        
        if static:
            self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        else:
            self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        
        self.body.velocity = (0, 0)
        self.body.position = pos
        
        self.shape = shape_constructor(self.body, params)
        self.shape.collision_type = col_type
        self.shape.elasticity = 1.
        self.shape.density = 30.
        system.space.add(self.body, self.shape)
        self.system = system
        self.color = color
        self.shape_type = shape_type
        self.display_type = display_type
    
    def draw(self):
        x, y = self.body.position
        x = x - self.system.camera_pos[0]
        y = y - self.system.camera_pos[1]
        _, h = pygame.display.get_surface().get_size()
        y = h - y
        
        
        if self.display_type == 'square':
            surf = pygame.Surface(size=self.disp_size)
            surf.set_colorkey(colors.BLACK)
            surf.fill(color=self.color)
            surf = pygame.transform.rotate(surf, self.body.angle * (180.0 / math.pi))
            surf_size = surf.get_size()
            x = x - (surf_size[0] / 2)
            y = y - (surf_size[1] / 2)
            self.system.display.blit(surf, (x, y))
        elif self.display_type == 'triangle':
            vertices = get_triangle_vertices(self.disp_size, -self.body.angle)
            for i in range(len(vertices)):
                vertices[i] = add_vec2tuple((x, y), vertices[i])
            pygame.draw.polygon(self.system.display, self.color, vertices)
        elif self.display_type == 'circle':
            pygame.draw.circle(self.system.display, self.color, (x, y), self.disp_size)
    
    def update(self, deltaTime):
        pass
