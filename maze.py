import numpy as np
import random
import colors
from create_gameobject import create_gameobject

GOAL_COLLISION_TYPE = 3

directions = [
    np.array([1, 0], dtype=int), 
    np.array([0, 1], dtype=int), 
    np.array([-1, 0], dtype=int), 
    np.array([0, -1], dtype=int)]


def rank_directions():
    # random ranking of directions
    random.shuffle(directions)
    return directions


# each node has 4 walls
class Cell:
    
    def __init__(self):
        self.left = True
        self.right = True
        self.up = True
        self.down = True
        self.visited = False



class Maze:
    
    def __init__(self, width, height):
        self.maze = [[Cell() for _ in range(height)] for _ in range(width)]
        self.pos = np.zeros(shape=(2,), dtype=int)
        self.width = width
        self.height = height
        self.stack = []
        self.visited_stack = []
    
    def in_range(self, pos):
        not_over = pos[0] < self.width and pos[1] < self.height
        not_under = np.min(pos) >= 0
        return not_over and not_under
    
    def print_maze(self):
        for i_ in range(self.height):
            for j in range(self.width):
                i = self.height - i_ - 1
                cell = self.maze[j][i]
                if cell.left:
                    print("|", sep="", end="")
                else:
                    print(" ", sep="", end="")
                
                if cell.down:
                    print("_", sep="", end="")
                else:
                    print(" ", sep="", end="")
                
                if cell.right:
                    print("|", sep="", end="")
                else:
                    print(" ", sep="", end="")
                
            print("")
    
    
    def move_to(self, pos, direction):
        new_pos = pos + direction
        if self.in_range(new_pos) and \
                not self.maze[new_pos[0]][new_pos[1]].visited:
            if direction[0] == 1 and direction[1] == 0:
                self.maze[pos[0]][pos[1]].right = False
                self.maze[new_pos[0]][new_pos[1]].left = False
                self.pos = new_pos
                self.maze[new_pos[0]][new_pos[1]].visited = True
                self.visited_stack.append(new_pos)
                return True
            elif direction[0] == -1 and direction[1] == 0:
                self.maze[pos[0]][pos[1]].left = False
                self.maze[new_pos[0]][new_pos[1]].right = False
                self.pos = new_pos
                self.maze[new_pos[0]][new_pos[1]].visited = True
                self.visited_stack.append(new_pos)
                return True
            elif direction[0] == 0 and direction[1] == 1:
                self.maze[pos[0]][pos[1]].up = False
                self.maze[new_pos[0]][new_pos[1]].down = False
                self.pos = new_pos
                self.maze[new_pos[0]][new_pos[1]].visited = True
                self.visited_stack.append(new_pos)
                return True
            elif direction[0] == 0 and direction[1] == -1:
                self.maze[pos[0]][pos[1]].down = False
                self.maze[new_pos[0]][new_pos[1]].up = False
                self.pos = new_pos
                self.maze[new_pos[0]][new_pos[1]].visited = True
                self.visited_stack.append(new_pos)
                return True
        return False
    
    def top_stack(self):
        directions = rank_directions()
        for dir in directions:
            new_pos = self.pos + dir
            if self.in_range(new_pos):
                if not self.maze[new_pos[0]][new_pos[1]].visited:
                    self.stack.append((self.pos, dir))
    
    def gen_random_maze(self):
        moved = True
        self.pos[0] = random.randint(0, self.width-1)
        self.pos[1] = random.randint(0, self.height-1)
        self.top_stack()
        while len(self.stack) > 0:
            if moved:
                self.top_stack()
            pos, dir = self.stack.pop()
            moved = self.move_to(pos, dir)
    
    def extract_maze_edges(self):
        width, height = self.width, self.height
        edges = [None for _ in range(width*2 + 1)]
        for i_ in range(width):
            i = i_ * 2
            edges[i] = [True for _ in range(height)]
            edges[i+1] = [True for _ in range(height+1)]
        edges[-1] = [True for _ in range(height)]
        
        for i_ in range(height):
            for j_ in range(width):
                i = height - i_ - 1
                j = j_ * 2
                if not self.maze[j_][i].left:
                    edges[j][i] = False
                if not self.maze[j_][i].up:
                    edges[j+1][i+1] = False
                if not self.maze[j_][i].down:
                    edges[j+1][i] = False
                if not self.maze[j_][i].right:
                    edges[j+2][i] = False
        
        return edges


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



def create_maze(game, maze_size, dim, x_pos=0, y_pos=0):
    edge_width = 4
    maze = Maze(dim, dim)
    maze.gen_random_maze()
    edges = maze.extract_maze_edges()
    
    cell_size = maze_size / dim
    cell_size_half = cell_size / 2
    game.maze_cell_size = cell_size
    game.maze_grid_size = dim
    
    rx, ry, xg, yg = random_maze_edge(dim, dim)
    rx = 1
    ry = -1
    edges[rx][ry] = False
    
    goal_x = x_pos + cell_size_half + (xg * cell_size)
    goal_y = y_pos + cell_size_half + (yg * cell_size)
    
    if False:
        # place flag
        create_gameobject(game,
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
                create_gameobject(game, x_pos + x, y_pos + y, edge_width, cell_size, color=color, static=True)
        
        # draw top and bottom sides
        for j in range(len(edges[i+1])):
            if edges[i+1][j]:
                x = i_ * cell_size + cell_size_half
                y = j * cell_size
                color = (0, 1.0 * 255, float(i) * 255.0 / len(edges))
                color = colors.WHITE
                create_gameobject(game, x_pos + x, y_pos + y, cell_size, edge_width, color=color, static=True)
    
    # draw right sides
    for j in range(len(edges[-1])):
        if edges[-1][j]:
            x = maze_size
            y = j * cell_size + cell_size_half
            color = colors.CYAN
            color = colors.WHITE
            create_gameobject(game,
                x_pos + x, y_pos + y, edge_width, cell_size, 
                color, static=True)


