import cv2
from enum import Enum

class FrameType(Enum):
    YUV = 1
    RGB = 2

class Frame():
    def __init__(self, frame, frame_number=-1, frame_type = FrameType.YUV):
        self.frame = frame
        self.height, self.width, _ = frame.shape
        self.frame_type = frame_type

        self.frame_number = frame_number

        if self.frame_type == FrameType.YUV:
            self.y = frame[:, :, 0]
            self.u = frame[:, :, 1]
            self.v = frame[:, :, 2]

    def set_y(self, new_y):
        self.frame[:, :, 0] = new_y
        self.y = self.frame[:, :, 0]   # optional, keeps the view explicit


class Video_IO():
    def __init__(self, video_path, yuv=True):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if yuv:
            self.frame_type = FrameType.YUV
        else:
            self.frame_type = FrameType.RGB

        self.frame_number = -1


    def read_frame(self):

        ret, frame = self.cap.read()

        if self.frame_type == FrameType.YUV:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        if not ret:
            return None
        self.frame_number += 1
        return Frame(frame, frame_type=self.frame_type, frame_number=self.frame_number)

    def release(self):
        self.cap.release()
        # The writer is created lazily by write_frame, so it may not exist. Releasing it
        # here is what flushes the moov atom: without this the output mp4 stays
        # unreadable until the interpreter exits and GC finalises it, which breaks any
        # caller that writes a video and then reads it back in the same process.
        out = getattr(self, "out", None)
        if out is not None:
            out.release()
            self.out = None

    def write_frame(self, frame, output_path):
        if frame.frame_number == 0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        if self.frame_type == FrameType.YUV:
            frame_to_write = cv2.cvtColor(frame.frame, cv2.COLOR_YUV2BGR)
        else:
            frame_to_write = frame.frame

        self.out.write(frame_to_write)
