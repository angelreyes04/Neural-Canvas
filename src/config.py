import torch 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
IMG_SIZE = 512 
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 1e6
LEARNING_RATE = 0.02
NUM_STEPS = 500

# Paths
CONTENT_IMAGE_PATH = ""
STYLE_IMAGE_PATHS = [""]  
OUTPUT_IMAGE_PATH = ""