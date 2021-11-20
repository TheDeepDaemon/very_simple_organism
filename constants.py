


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


INPUT_SHAPE = (16, 16, 1)
INPUT_SIZE = 1
for dim in INPUT_SHAPE:
    INPUT_SIZE *= dim
NUM_ACTIONS = 3

# the number of "memories" to use for training
NUM_MEMORIES = 1000

# short term memory size.
# this should be at least the number
# of the highest derivative of motion
# the network should be able to detect
ST_MEM_SIZE = 4
PREDICTION_FRAMES = 4

# the size of the agent's internal map
MAP_SIZE = 64

GRANULARITY = 64

HIDDEN_FILTERS = 64
