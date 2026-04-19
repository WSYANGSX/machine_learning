import cv2
import numpy as np

from machine_learning.utils.streams import VideoStream, WebcamStream


# # video
# video = VideoStream({"path1": "/home/yangxf/Downloads/1.mp4", "path2": "/home/yangxf/Downloads/2.mp4"}, fps=60)
# print(video.valid_frames, video.fps, video.width, video.height)

# delay = max(1, int(1000 / video.fps))

# for frame in video:
#     if isinstance(frame, dict):
#         frame1 = frame["path1"]
#         frame2 = frame["path2"]
#         frame = np.hstack([frame1, frame2])
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(delay) & 0xFF == ord("q"):
#         break

# video.release()
# cv2.destroyAllWindows()

# camera
video = WebcamStream(0, width=1344, height=376, fps=60)
print(video.frames, video.fps, video.width, video.height)


for frame in video:
    frame_left, frame_right = np.split(frame, 2, axis=1)

    cv2.imshow("Camera 0 (Left/RGB1)", frame_left)
    cv2.imshow("Camera 1 (Right/RGB2)", frame_right)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
