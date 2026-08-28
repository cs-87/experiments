import subprocess
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import shutil

# Rate control for every video this module writes. CRF is a quality target, not a
# bitrate, so the same number means the same amount of coding damage across clips of
# different content -- which is the property a robustness sweep needs and the one thing
# OpenCV's writer cannot give.
DEFAULT_CRF = 23
DEFAULT_PRESET = "medium"


class FFmpegWriter:
    """
    H.264 writer at a fixed CRF, driven by raw BGR frames over a pipe.

    cv2.VideoWriter with 'mp4v' is MPEG-4 Part 2 at whatever bitrate OpenCV happens to
    pick -- measured between 3 and 20 Mbps on the 1080p clips here, varying with content
    and with the mark itself. A watermark that survives at 20 Mbps and dies at 3 reads as
    a property of the watermark when it is a property of the encoder, so every sweep run
    has to be encoded at one fixed quality or the comparison means nothing.

    Written through a pipe rather than by transcoding a temporary file afterwards: a
    second generation of encoding would be an extra, uncontrolled attack sitting between
    the embedder and everything downstream of it.
    """

    def __init__(self, path, width, height, fps, crf=DEFAULT_CRF,
                 preset=DEFAULT_PRESET, pix_fmt="yuv420p"):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH; needed by FFmpegWriter")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.width, self.height = int(width), int(height)
        # A container needs a real frame rate; cv2 reports 0 for some sources.
        self.fps = float(fps) if fps and fps > 0 else 30.0

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", f"{self.fps}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", pix_fmt,
            self.path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, bgr_frame):
        if bgr_frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame is {bgr_frame.shape[:2]}, writer was opened for "
                f"{(self.height, self.width)}"
            )
        self.proc.stdin.write(np.ascontiguousarray(bgr_frame, dtype=np.uint8).tobytes())

    def release(self):
        if self.proc is None:
            return
        # Closing stdin is what tells ffmpeg the stream ended; without the wait() the
        # file's moov atom may not be written by the time a caller reads it back.
        self.proc.stdin.close()
        code = self.proc.wait()
        self.proc = None
        if code != 0:
            raise RuntimeError(f"ffmpeg exited {code} while writing {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


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
    def __init__(self, video_path, yuv=True, codec="h264", crf=DEFAULT_CRF,
                 preset=DEFAULT_PRESET):
        """
        codec: "h264" writes through FFmpegWriter at a fixed `crf` -- the default,
        because a sweep needs its encode held constant. "mp4v" restores the old
        cv2.VideoWriter path for anything that has to reproduce an earlier output.
        """
        self.video_path = video_path
        self.codec = codec
        self.crf = crf
        self.preset = preset
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

        # "h264" pipes to ffmpeg at a fixed CRF (see FFmpegWriter); "mp4v" keeps the old
        # cv2.VideoWriter path for callers that only want a file to look at.
        self.codec = codec
        self.crf = crf
        self.out = None


    def read_frame(self):

        ret, frame = self.cap.read()

        # The guard has to come first: at end of stream cap.read() hands back
        # (False, None), and cvtColor on None raises cv2.error instead of letting
        # the caller see the None it is checking for.
        if not ret:
            return None
        if self.frame_type == FrameType.YUV:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
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
        # Opened on the first frame written, not on frame_number == 0: a caller that
        # strides, starts mid-clip, or feeds frames it built itself never presents a
        # frame 0, and the old test left self.out as None for the whole run.
        if self.out is None:
            if self.codec == "mp4v":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(
                    output_path, fourcc, self.fps, (self.width, self.height))
            else:
                self.out = FFmpegWriter(
                    output_path, self.width, self.height, self.fps,
                    crf=self.crf, preset=self.preset)

        if self.frame_type == FrameType.YUV:
            frame_to_write = cv2.cvtColor(frame.frame, cv2.COLOR_YUV2BGR)
        else:
            frame_to_write = frame.frame

        self.out.write(frame_to_write)
