from game_object import GameObject


def create_gameobject(game, x, y, width, height, color, collision_type=1, static=False):
        obj = GameObject(
            game, (x, y), 
            (width, height), 
            col_type=collision_type, 
            color=color, static=static)
        game.game_objects.append(obj)
        return obj
