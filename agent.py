import colors
from game_object import GameObject
from brain import AgentBrain


# interface between the AI and the environment
# this is basically a game object with a brain
class Agent(GameObject):
    
    def __init__(self, system, pos, size, col_type):
        super().__init__(
            system, pos, size, col_type, 
            colors.RED, static=False, 
            shape_type='circle', display_type='triangle')
        self.brain = AgentBrain()
    
    def draw(self):
        super().draw()
        

