from cv2 import circle
from numpy.core.fromnumeric import shape, size
import pygame
from pygame import display
import pymunk
import colors
import math

# point from a circle given a circle radius and an angle
def point_from_circle(radius, angle):
    return (radius * math.cos(angle), radius * math.sin(angle))

def get_triangle_vertices(radius, angle):
    v1 = point_from_circle(radius, angle)
    v2 = point_from_circle(radius, angle + (math.pi * 2.0 / 3.0))
    v3 = point_from_circle(radius, angle + (math.pi * 4.0 / 3.0))
    return [v1, v2, v3]



def add_vec2tuple(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1])

def make_square_tuple(size):
    if type(size) is int:
        return (size, size)
    else:
        return size



def create_box(body, params):
    return pymunk.Poly.create_box(body, params)

def create_triangle(body, params):
    rad = params
    return pymunk.Poly(body, get_triangle_vertices(rad, 0))

def create_circle(body, radius):
    return pymunk.Circle(body, radius=radius)


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
