from typing import Iterator, Union, List

import os
import cv2
import time
import threading
import numpy as np

from machine_learning.utils.logger import LOGGER


class VideoStream:
    def __init__(
        self,
        path: str,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ):
        """Load video from a local file with optional dynamic resizing and downsampling."""
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

        # video original properties
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.orig_fps = self.cap.get(cv2.CAP_PROP_FPS)

        orig_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.orig_frames = int(orig_frames) if orig_frames > 0 else float("inf")

        # output properties
        self.width = width if width else self.orig_width
        self.height = height if height else self.orig_height
        self.fps = fps if fps else self.orig_fps

        # Safely calculate the frame extraction interval
        if self.fps is not None and self.orig_fps > 0 and self.fps < self.orig_fps:
            self.stride = max(1, int(round(self.orig_fps / self.fps)))
        else:
            self.stride = 1

        self.count = 0

        LOGGER.info(
            f"Loaded Video: '{os.path.basename(self.path)}' | "
            f"Original: {self.orig_width}x{self.orig_height} @ {self.orig_fps:.2f} FPS, {self.orig_frames} frames | "
            f"Output: {self.width}x{self.height} @ {self.fps} FPS (Skip interval: {self.stride - 1})"
        )

    @property
    def frames(self) -> int:
        return len(self)

    def __iter__(self) -> Iterator[np.ndarray]:
        self.count = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> np.ndarray:
        for _ in range(self.stride - 1):
            if self.count >= self.orig_frames:
                break
            ret = self.cap.grab()  # Only move the pointer, never retrieve

            if not ret:
                break
            self.count += 1

        if self.count >= self.orig_frames:
            self.release()
            raise StopIteration

        ret, frame = self.cap.read()  # grab() + retrieve()
        if not ret:
            self.release()
            raise StopIteration

        self.count += 1

        # resize
        if self.width is not None and self.height is not None:
            if (self.width, self.height) != (self.orig_width, self.orig_height):
                frame = cv2.resize(frame, (self.width, self.height))

        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __len__(self) -> int:
        return int(self.orig_frames // self.stride) if self.orig_frames != float("inf") else 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class WebcamStream:
    def __init__(
        self,
        sources: Union[int, str, List[Union[int, str]]],
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ):
        """
        Load live stream from one or multiple webcams/RTSP streams for multi-modal inference.
        Supports software resizing and FPS downsampling (throttling).
        """
        if not isinstance(sources, (list, tuple)):
            sources = [sources]

        self.sources = sources
        self.num_cams = len(sources)
        self.caps = [cv2.VideoCapture(s) for s in sources]

        for s, cap in zip(self.sources, self.caps):
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open webcam/stream: {s}")

        # Set buffer size to 1 to reduce latency (clear OS buffer)
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Retrieve original properties from the primary camera (index 0)
        self.orig_width = int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.orig_fps = self.caps[0].get(cv2.CAP_PROP_FPS)
        if self.orig_fps <= 0 or np.isnan(self.orig_fps):
            self.orig_fps = 30.0  # Fallback for some webcams that return 0

        # Output properties
        self.width = width if width else self.orig_width
        self.height = height if height else self.orig_height
        self.fps = fps if fps else self.orig_fps

        # FPS Throttling for live stream
        # Instead of skipping frames, we compute the target delay between yields
        self.target_delay = 1.0 / self.fps if self.fps < self.orig_fps else 0.0
        self.last_yield_time = 0.0

        # Shared variables for threading
        self.imgs = [None] * self.num_cams
        self.running = True

        # Start daemon thread to keep reading the latest frames
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

        # Wait until the first frame is successfully captured by all cameras
        while not all(img is not None for img in self.imgs) and self.running:
            time.sleep(0.01)

        sources_str = ", ".join([str(s) for s in self.sources])
        LOGGER.info(
            f"Loaded Webcam: [{sources_str}] | "
            f"Original: {self.orig_width}x{self.orig_height} @ {self.orig_fps:.2f} FPS | "
            f"Output: {self.width}x{self.height} @ {self.fps} FPS"
        )

    @property
    def frames(self) -> int:
        return float("inf")

    def _update(self):
        """Background thread to read frames continuously to avoid buffer accumulation."""
        # Polling slightly faster than hardware FPS to ensure the buffer is always drained
        poll_interval = 1.0 / (self.orig_fps * 1.5)

        while self.running:
            # Step 1: Grab all cameras simultaneously for hardware-level sync
            for cap in self.caps:
                cap.grab()

            # Step 2: Retrieve the decoded frames
            for i, cap in enumerate(self.caps):
                ret, frame = cap.retrieve()
                if ret:
                    self.imgs[i] = frame

            time.sleep(poll_interval)  # yield thread

    def __iter__(self) -> Iterator[Union[np.ndarray, List[np.ndarray]]]:
        return self

    def __next__(self) -> Union[np.ndarray, List[np.ndarray]]:
        if not self.running:
            raise StopIteration

        if self.target_delay > 0:
            elapsed = time.time() - self.last_yield_time
            wait_time = self.target_delay - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        self.last_yield_time = time.time()

        # Return a copy of the latest frames
        frames = [img.copy() for img in self.imgs]

        if (self.width, self.height) != (self.orig_width, self.orig_height):
            frames = [cv2.resize(f, (self.width, self.height)) for f in frames]

        # If single camera, return the array directly; if multi-modal, return the list
        return frames[0] if self.num_cams == 1 else frames

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        for cap in self.caps:
            if cap.isOpened():
                cap.release()

    def __len__(self) -> float:
        return float("inf")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
