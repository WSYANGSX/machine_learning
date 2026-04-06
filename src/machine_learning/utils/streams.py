from typing import Iterator, Union, List

import os
import cv2
import time
import threading
import numpy as np


class VideoStream:
    def __init__(self, path: str):
        """Load video from a local file."""
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.count = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        self.count = 0
        return self

    def __next__(self) -> np.ndarray:
        if self.count >= self.frames:
            self.release()
            raise StopIteration

        ret, frame = self.cap.read()
        if not ret:
            self.release()
            raise StopIteration

        self.count += 1

        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __len__(self) -> int:
        return self.frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class WebcamStream:
    def __init__(self, sources: Union[int, str, List[int | str]]):
        """
        Load live stream from one or multiple webcams/RTSP streams for multi-modal inference.
        """
        if not isinstance(sources, (list, tuple)):
            sources = [sources]

        self.sources = sources
        self.num_cams = len(sources)
        self.caps = [cv2.VideoCapture(s) for s in sources]

        for s, cap in zip(self.sources, self.caps):
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open webcam/stream: {s}")

        # Set buffer size to 1 to reduce latency
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.fps = self.caps[0].get(cv2.CAP_PROP_FPS) or 30.0

        # Shared variables for threading
        self.imgs = [None] * self.num_cams
        self.running = True

        # Start daemon thread to keep reading the latest frames
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

        # Wait until the first frame is successfully captured by all cameras
        while not all(img is not None for img in self.imgs) and self.running:
            time.sleep(0.01)

    def _update(self):
        """Background thread to read frames continuously to avoid buffer accumulation."""
        while self.running:
            # Step 1: Grab all cameras simultaneously to hardware-sync their timestamps
            # This is critical for multi-modal (e.g., RGB + IR) synchronization!
            for cap in self.caps:
                cap.grab()

            # Step 2: Retrieve the decoded frames
            for i, cap in enumerate(self.caps):
                ret, frame = cap.retrieve()
                if ret:
                    self.imgs[i] = frame

            time.sleep(1 / self.fps)  # yield thread

    def __iter__(self) -> Iterator[Union[np.ndarray, List[np.ndarray]]]:
        return self

    def __next__(self) -> Union[np.ndarray, List[np.ndarray]]:
        if not self.running:
            raise StopIteration

        # Return a copy of the latest frames to avoid being overwritten by the thread
        frames = [img.copy() for img in self.imgs]

        # If single camera, return the array directly; if multi-modal, return the list
        return frames[0] if self.num_cams == 1 else frames

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        for cap in self.caps:
            if cap.isOpened():
                cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
