from typing import Iterator, Union, List

import os
import cv2
import time
import threading
import numpy as np

from machine_learning.utils.logger import LOGGER


class StreamBase:
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoStream(StreamBase):
    def __init__(
        self,
        path: Union[str, dict[str, str]],
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ):
        """
        Load video from one or multiple local files (for multi-modal inference)
        with optional dynamic resizing and downsampling.
        """
        self.input_is_dict = isinstance(path, dict)
        if not self.input_is_dict:
            paths_dict = {"img": path}
        else:
            paths_dict = path

        self.keys = list(paths_dict.keys())
        self.paths = [os.path.abspath(p) for p in paths_dict.values()]
        self.num_vids = len(self.paths)
        self.caps = []

        for p in self.paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Video file not found: {p}")

            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {p}")
            self.caps.append(cap)

        # Obtain the original attributes
        self.orig_widths = [int(self.caps[i].get(cv2.CAP_PROP_FRAME_WIDTH)) for i in range(self.num_vids)]
        self.orig_heights = [int(self.caps[i].get(cv2.CAP_PROP_FRAME_HEIGHT)) for i in range(self.num_vids)]
        self.orig_fpses = [cap.get(cv2.CAP_PROP_FPS) for cap in self.caps]
        self.orig_frames = [cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in self.caps]

        # vaild attributes
        self.valid_fps = min(self.orig_fpses)
        self.valid_frames = int(min(self.orig_frames)) if min(self.orig_frames) > 0 else float("inf")

        # output properties
        self.width = width if width else self.orig_widths[0]
        self.height = height if height else self.orig_heights[0]

        if fps is not None:
            self.fps = min(fps, self.valid_fps)
            if fps > self.valid_fps:
                LOGGER.warning(
                    f"Requested FPS ({fps}) exceeds original video FPS ({self.valid_fps:.2f}). "
                    f"Capped to {self.fps:.2f} FPS."
                )
        else:
            self.fps = self.valid_fps

        # Safely calculate the frame extraction interval
        if self.fps is not None and self.valid_fps > 0 and self.fps < self.valid_fps:
            self.stride = max(1, int(round(self.valid_fps / self.fps)))
        else:
            self.stride = 1

        self.count = 0

        paths_str = ", ".join([f"{k}: {os.path.basename(p)}" for k, p in zip(self.keys, self.paths)])
        orig_str = " | ".join(
            [
                f"{k} ({self.orig_widths[i]}x{self.orig_heights[i]} @ {self.orig_fpses[i]:.2f} FPS,"
                f" {self.orig_frames[i]} frames)"
                for i, k in enumerate(self.keys)
            ]
        )

        LOGGER.info(
            f"Loaded Video(s): {{{paths_str}}} | "
            f"Original: {orig_str} | "
            f"Output: {self.width}x{self.height} @ {self.fps} FPS (Skip interval: {self.stride - 1})"
        )

    @property
    def frames(self) -> int:
        return len(self)

    def __iter__(self) -> Iterator[Union[np.ndarray, dict[str, np.ndarray]]]:
        self.count = 0
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> Union[np.ndarray, dict[str, np.ndarray]]:
        for _ in range(self.stride - 1):
            if self.count >= self.valid_frames:
                break

            rets = [cap.grab() for cap in self.caps]
            if not all(rets):
                break
            self.count += 1

        if self.count >= self.valid_frames:
            self.release()
            raise StopIteration

        frames = []
        for cap in self.caps:
            ret, frame = cap.read()
            if not ret:
                self.release()
                raise StopIteration
            frames.append(frame)

        self.count += 1

        if self.width is not None and self.height is not None:
            for i, frame in enumerate(frames):
                if (self.width, self.height) != (self.orig_widths[i], self.orig_heights[i]):
                    frames[i] = cv2.resize(frame, (self.width, self.height))

        return {k: f for k, f in zip(self.keys, frames)} if self.input_is_dict else frames[0]

    def release(self):
        for cap in self.caps:
            if cap.isOpened():
                cap.release()

    def __len__(self) -> int:
        return int(self.valid_frames // self.stride) if self.valid_frames != float("inf") else 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class WebcamStream(StreamBase):
    def __init__(
        self,
        sources: Union[int, str, dict[str, Union[int, str]]],
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ):
        """
        Load live stream from one or multiple webcams/RTSP streams for multi-modal inference.
        """
        self.input_is_dict = isinstance(sources, dict)
        if not self.input_is_dict:
            sources_dict = {"img": sources}
        else:
            sources_dict = sources

        self.keys = list(sources_dict.keys())
        self.sources = list(sources_dict.values())
        self.num_cams = len(self.sources)
        self.caps = [cv2.VideoCapture(s) for s in self.sources]

        for s, cap in zip(self.sources, self.caps):
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open webcam/stream: {s}")

        # Set buffer size to 1 to reduce latency (clear OS buffer)
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps is not None:
                cap.set(cv2.CAP_PROP_FPS, fps)

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

        sources_str = ", ".join([f"{k}: {s}" for k, s in zip(self.keys, self.sources)])
        LOGGER.info(
            f"Loaded Webcam: {{{sources_str}}} | "
            f"Original: {self.orig_width}x{self.orig_height} @ {self.orig_fps:.2f} FPS | "
            f"Output: {self.width}x{self.height} @ {self.fps} FPS"
        )

    @property
    def frames(self) -> int:
        return float("inf")

    def _update(self):
        """Background thread to read frames continuously to avoid buffer accumulation."""
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

    def __iter__(self) -> Iterator[Union[np.ndarray, dict[str, np.ndarray]]]:
        return self

    def __next__(self) -> Union[np.ndarray, dict[str, np.ndarray]]:
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

        return {k: f for k, f in zip(self.keys, frames)} if self.input_is_dict else frames[0]

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
