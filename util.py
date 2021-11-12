import math


# point from a circle given a circle radius and an angle
def point_from_circle(radius, angle):
    return (radius * math.cos(angle), radius * math.sin(angle))


# get an array of points that form a triangle
def get_triangle_vertices(radius, angle):
    v1 = point_from_circle(radius, angle)
    v2 = point_from_circle(radius, angle + (math.pi * 2.0 / 3.0))
    v3 = point_from_circle(radius, angle + (math.pi * 4.0 / 3.0))
    return [v1, v2, v3]


# add tuples or vecs, output tuple
def add_vec2tuple(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1])


# make a tuple out of a single number
def make_square_tuple(size):
    if type(size) is int:
        return (size, size)
    else:
        return size
