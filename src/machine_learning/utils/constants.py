import os

NUM_THREADS = min(32, os.cpu_count() + 4)  # number of multiprocessing threads

# PATH
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
