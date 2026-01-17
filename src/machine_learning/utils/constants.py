import os

# I/O
NUM_THREADS = min(32, os.cpu_count() + 4)  # number of multiprocessing threads

# PATH
ROOT_PATH = "/home/yangxf/WorkSpace/machine_learning"
DATACFG_PATH = "/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/datasets"  # data cfg dir
ALGOCFG_PATH = "/home/yangxf/WorkSpace/machine_learning/src/machine_learning/cfg/algorithms"


# Img file types
IMG_FORMATS = [
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".pbm",
    ".pgm",
    ".ppm",
    ".sr",
    ".ras",
    ".tiff",
    ".tif",
    ".webp",
]

NPY_FORMATS = [".npy", ".npz"]

CSS_COLORS = [
    "red",
    "green",
    "blue",
    "cyan",
    "magenta",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "lightblue",
    "darkgreen",
    "navy",
    "teal",
    "lime",
    "olive",
    "maroon",
    "silver",
    "gold",
    "violet",
]
