import random

tracked_colors = {}

def get_color_for_id(track_id):
    """
    Generates or returns a unique, consistent color for a given object ID.
    """
    if track_id not in tracked_colors:
        random.seed(int(track_id))
        tracked_colors[track_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return tracked_colors[track_id]
