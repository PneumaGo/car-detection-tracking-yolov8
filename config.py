import os

# Counting line
COUNTING_LINE_Y = 500
COUNTING_LINE_START_X = 0
COUNTING_LINE_END_X = 1280

# Object history
MAX_HISTORY_LENGTH = 10

# Paths
MODEL_PATH = os.path.join("models", "best.pt")
TEMP_OUTPUT_IMAGE = "temp_annotated_image.jpg"
TEMP_OUTPUT_VIDEO = "temp_output_video.mp4"