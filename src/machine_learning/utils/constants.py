import os

NUM_THREADS = min(32, os.cpu_count() + 4)  # number of YOLO multiprocessing threads
